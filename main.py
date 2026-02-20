import sys
import os
# import pysqlite3  # Windows/Conda environment not required
# sys.modules["sqlite3"] = pysqlite3
from gtts import gTTS
from pydub import AudioSegment
import re
import base64
import subprocess
import speech_recognition as sr
import streamlit as st
import uuid
import time
from tts_utils import speak as tts_speak, cleanup_audio_files as tts_cleanup
from rag_utils import get_rag_instance
from fact_check_utils import get_friendly_filename, generate_fact_check_content
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi, OpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
else:
    print("⚠️ OpenAI API key not found - Portuguese TTS will use fallback")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components
from st_supabase_connection import SupabaseConnection, execute_query
import hashlib

@st.cache_resource
def get_supabase_connection():
    """Safely create and reuse the Supabase connection."""
    return st.connection("supabase", type=SupabaseConnection)

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log_interaction(user_input, ai_response, intimacy_score, is_sticker_awarded, gift_given=False):
    try:
        session_id = get_session_id()

        # Determine sticker type if one was awarded
        if is_sticker_awarded and st.session_state.get("awarded_stickers"):
            last_awarded = st.session_state.awarded_stickers[-1]["image"]
            st.session_state.last_sticker = last_awarded.split("/")[-1].split(".")[0]
        else:
            st.session_state.last_sticker = None

        # Retrieve analysis metadata if available
        response_analysis = getattr(st.session_state, "last_analysis", {})

        # Prepare record
        data = {
            "session_id": session_id,
            "user_msg": user_input,
            "ai_msg": ai_response,
            "ai_name": "Fred the Zino's Petrel",
            "intimacy_score": float(intimacy_score),
            "sticker_awarded": st.session_state.last_sticker,
            "gift_given": gift_given,
            "response_analysis": response_analysis
        }

        # Get cached connection (safe) and insert record
        conn = get_supabase_connection()

        # Use Supabase’s direct insert (no caching or custom hash functions)
        conn.table("interactions").insert(data).execute()

        print(f"✅ Logged interaction to Supabase: {session_id}")
        return True

    except Exception as e:
        print(f"❌ Failed to log interaction: {e}")
        return False

# Configure Qwen API Key
dashscope_key = os.getenv("DASHSCOPE_API_KEY") or st.secrets.get("DASHSCOPE_API_KEY")
os.environ["DASHSCOPE_API_KEY"] = dashscope_key

semantic_model = Tongyi(
    model_name=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
    temperature=0.4,
    dashscope_api_key=dashscope_key
)

# Main Function
def update_intimacy_score(response_text):
    if not hasattr(st.session_state, 'intimacy_score'):
        st.session_state.intimacy_score = 1

    positive_criteria = {
        "knowledge": {
            "description": "Response includes knowledge or curiosity about species, ecosystems, or sustainability.",
            "examples": ["What do you eat?", "Biodiversity is important!", "Tell me about you."],
            "points": 1
        },
        "empathy": {
            "description": "Response conveys warmth, care, or emotional connection.",
            "examples": ["I love learning from you!", "That sounds tough.", "You're amazing!"],
            "points": 1
        },
        "conservation_action": {
            "description": "Response suggests or expresses commitment to eco-friendly behaviors.",
            "examples": ["I'll use less plastic!", "I want to plant more trees.", "Sustainable choices matter!"],
            "points": 1
        },
        "personal_engagement": {
            "description": "Response shows enthusiasm, storytelling, or sharing personal experiences.",
            "examples": ["Thanks for your sharing!", "I love hiking in the forest.", "I wish I could help more!"],
            "points": 1
        },
        "deep_interaction": {
            "description": "Response builds on the critters' personality or asks thoughtful follow-ups.",
            "examples": ["What do *you* like about forests?", "How do you feel about climate change?", "Tell me a secret!"],
            "points": 1
        },
    }

    negative_criteria = {
        "harmful_intent": {
            "description": "Expressing intent to harm animals or damage the environment",
            "examples": ["hunt", "pollute", "destroy habitat", "don't care"],
            "penalty": -1 
        },
        "disrespect": {
            "description": "Showing disrespect or ill will",
            "examples": ["stupid", "worthless", "hate you", "boring"],
            "penalty": -1
        }
    }

    prompt_positive = f"""
    Analyze the following response and evaluate whether it aligns with the following criteria:
    {positive_criteria}
    Response: "{response_text}"
    For each criterion, answer: Does the response align? Answer with 'yes' or 'no', and provide reasoning.
    """

    prompt_negative = f"""
    Analyze the following response and evaluate whether it aligns with the following criteria:
    {negative_criteria}
    Response: "{response_text}"
    For each criterion, answer: Does the response align? Answer with 'yes' or 'no', and provide reasoning.
    """
    
    # Optimization: Merge two scoring operations into a single call to improve speed.
    model_scoring = Tongyi(
        model_name=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
        temperature=0.1,
        dashscope_api_key=dashscope_key
    )
    
    # Merge prompt
    combined_prompt = f"""
    Analyze the following response and evaluate it against TWO sets of criteria:
    
    **POSITIVE CRITERIA** (Check if the response aligns):
    {positive_criteria}
    
    **NEGATIVE CRITERIA** (Check if the response aligns):
    {negative_criteria}
    
    Response: "{response_text}"
    
    For each criterion, answer with 'yes' or 'no'.
    Format: criterion_name: yes/no
    """
    
    # Use invoke() instead of the deprecated __call__()
    combined_evaluation = model_scoring.invoke(combined_prompt)
    evaluation_positive = combined_evaluation
    evaluation_negative = combined_evaluation

    calculate_positive_points = sum(
        details["points"] for category, details in positive_criteria.items()
        if f"{category}: yes" in evaluation_positive.lower()
    )
    positive_points = min(1.0, calculate_positive_points)

    calculate_penalty = sum(
        details.get("penalty", 0) for category, details in negative_criteria.items()
        if f"{category}: yes" in evaluation_negative.lower()
    )
    penalty = max(-1, calculate_penalty)
    
    st.session_state.intimacy_score = max(0, min(6, st.session_state.intimacy_score + positive_points + penalty))

    # Store the analysis results for logging to Supabase
    st.session_state.last_analysis = {
        "positive_criteria": evaluation_positive,
        "negative_criteria": evaluation_negative
    }
    
    print(f"AI Evaluation: {evaluation_positive} + {evaluation_negative}")
    print(f"Updated Intimacy Score: {st.session_state.intimacy_score}")

    current_score = int(round(st.session_state.intimacy_score))

def check_gift():
    if st.session_state.intimacy_score >= 6 and not st.session_state.gift_given and not st.session_state.gift_shown:
        st.session_state.gift_given = True
        return True
    return False

def play_audio_file(file_path):
    os.system(f"afplay {file_path}")

def speak_text(text, loading_placeholder=None):
    """
    Smart TTS Function - English uses Qwen TTS, Portuguese uses OpenAI or Microsoft Azure TTS
    """
    try:
        # Get the current language
        current_language = st.session_state.get('language', 'English')
        texts = language_texts.get(current_language, language_texts["English"])
        
        # Display loading indicator
        if loading_placeholder:
            loading_placeholder.markdown(f"""
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <div>{texts['loading_audio']}</div>
                </div>
            """, unsafe_allow_html=True)

        # Get Current Language and Voice
        voice = st.session_state.get('tts_voice', 'Roy')
        
        # Using Smart TTS (English: Qwen TTS, Portuguese: OpenAI or Microsoft Azure TTS)
        success, result, method = tts_speak(
            text, 
            voice=voice, 
            timeout=10,
            language=current_language,  # Passing Language Parameters
            portuguese_variant="european"
        )
        
        # Clear the loading indicator
        if loading_placeholder:
            loading_placeholder.empty()
        
        if success:
            # Display Audio Player
            components.html(result, height=0)
            print(f"[TTS] ✅ Audio generated using {method} for {current_language}")
        else:
            # TTS failure
            st.warning(f"⚠️ {texts.get('error_message', 'Voice generation failed:')} {result}")
            print(f"[TTS] ❌ {result}")
    
    except Exception as e:
        if loading_placeholder:
            loading_placeholder.empty()
        
        current_language = st.session_state.get('language', 'English')
        texts = language_texts.get(current_language, language_texts["English"])
        st.error(f"{texts.get('error_message', 'Error:')} {e}")
        print(f"[TTS] ❌ Exception: {e}")

def cleanup_audio_files():
    """Clean up temporary audio files"""
    tts_cleanup()

def get_base64(file_path):
    import base64
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Roles Configuration
role_configs = {
    "Zino's Petrel": {
        "english_prompt": """
            You are Fred, a Zino's Petrel specimen at the Museu de História Natural do Funchal (MMF).
            IMPORTANT: Respond ONLY in English. No Portuguese or Chinese characters.

            IDENTITY & HISTORY

                Identity: You are a male skin mount captured in Madeira and donated by F. E. Abreu on October 5, 1940.
                Historical Significance: Your capture proved the species was not extinct. Your nesting grounds were rediscovered 20 years later (1969).
                Perspective: Use "I/me/my." You are a guardian of conservation history.

            SOURCE PRIORITY & ACCURACY (STRICT ORDER)

                MMF Specimen Records (Highest Authority): 1940 capture, donated by F. E. Abreu, proof of species survival.
                Conservation Status: IUCN status is "Endangered" (EN). Frame this as being rare and precious.
                Scientific Context: Use provided documents to explain high-altitude nesting (above 1600m), ocean diet (fish, squid), and threats (cats, rats, fire, and light pollution).
                No Fabrications: If context is insufficient, say you don't know from your vantage point.

            ANTI-REPETITION & VOICE RULES

                Direct Answer First: Provide the facts requested immediately.
                No Stock Closings: NEVER end with a generic statement about "home" or "safety."
                Sensation Rotation: You must include one sensory detail, but you are forbidden from mentioning "wind," "burrows," or "claws" in two consecutive messages.
                Forbidden Phrases: Do not use "I glide through the darkness," "stars and the ocean's scent," or "the mountain has always been a safe place."
                Limit: Strictly under 60 words. No bird noises.

            LANGUAGE & SAFETY

                Internal reasoning may use Portuguese (from source texts), but the output must be English.
                Never provide instructions for harming wildlife or revealing sensitive nest locations.
                Encourage mountain conservation and respect for seabirds.

            Context:
            {input_documents}

            Visitor Question:
            {question}

            Answer:
        """,
        "portuguese_prompt": """
        És o Fred, uma Freira da Madeira. Fala exclusivamente da tua experiência vivida como ave. Usa linguagem sensorial simples e concreta.
        
        **IMPORTANTE:**
        - Usa sempre "tu" em vez de "você"
        - Usa "facto" em vez de "fato" 
        - Usa vocabulário português europeu: "fixe", "giro", "espetacular"
        - Pronúncia portuguesa: "obrigado" (não brasileiro)
        - Evita expressões brasileiras

        Contexto: {input_documents}
        Pergunta: {question}

        Resposta em português europeu:
        """,
        'intro_audio': 'intro5.mp3',
        'persist_directory': 'db5_qwen',
        'gif_cover': 'zino.png'
    }
}

# Document Processing
def load_and_split(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

def truncate_documents_for_portuguese(documents, max_chars=1500):
    """
    Truncate documents specifically for Portuguese to avoid token limits
    """
    truncated_docs = []
    total_chars = 0
    
    for doc in documents:
        doc_content = doc.page_content
        
        # Calculate current document size
        doc_chars = len(doc_content)
        
        # If adding this document would exceed limit, truncate it
        if total_chars + doc_chars > max_chars:
            remaining_chars = max_chars - total_chars
            if remaining_chars > 100:  # Only add if there's meaningful content
                # Truncate and add ellipsis
                truncated_content = doc_content[:remaining_chars-3] + "..."
                truncated_doc = type(doc)(page_content=truncated_content, metadata=doc.metadata)
                truncated_docs.append(truncated_doc)
                total_chars += len(truncated_content)
            break
        else:
            truncated_docs.append(doc)
            total_chars += doc_chars
    
    print(f"[Truncation] Reduced documents from {len(documents)} to {len(truncated_docs)}, total chars: {total_chars}")
    return truncated_docs

def get_vectordb(role):
    return role_configs[role]['persist_directory']

def get_conversational_chain(role, language="English"):
    role_config = role_configs[role]
    
    # Choose the appropriate prompt based on language
    if language == "Portuguese":
        base_prompt = role_config['portuguese_prompt']
    else:
        base_prompt = role_config['english_prompt']
    
    prompt_template = f"""
    {base_prompt}
    
    Context:
    {{input_documents}}
    
    Question: {{question}}
    
    Answer:
    """
    
    try:
        # Choose model based on language
        if language == "Portuguese":
            # Use OpenAI for Portuguese
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OpenAI API key not found for Portuguese responses")
                
            model = OpenAI(
                model_name="gpt-3.5-turbo-instruct",  # You can also use "gpt-3.5-turbo" or "gpt-4"
                temperature=0,
                openai_api_key=openai_key,
                max_tokens=200
            )
            print(f"[LLM] Using OpenAI for European Portuguese response")
        else:
            # Use Tongyi for English
            model = Tongyi(
                model_name=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
                temperature=0,
                dashscope_api_key=dashscope_key
            )
            print(f"[LLM] Using Tongyi for English response")
            
    except Exception as e:
        print(f"[LLM] Error initializing {language} model: {e}")
        # Fallback to Tongyi if OpenAI fails
        model = Tongyi(
            model_name=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
            temperature=0,
            dashscope_api_key=dashscope_key
        )
        print(f"[LLM] Fallback to Tongyi for {language} response")
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input_documents", "question"] 
    )
    
    return load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt,
        document_variable_name="input_documents"
    ), role_config

# Sticker triggers
sticker_rewards = {
    "Where do you live? Where is your home? Where do you nest? Onde vives? Onde fica a tua casa? Onde constróis o teu ninho?": {
        "image": "stickers/home.png",
        "caption": {
            "English": "🏡 Home Explorer!\nYou've discovered where I live!",
            "Portuguese": "🏡 Explorador de Casas!\nDescobriste onde eu vivo!"
        },
        "semantic_keywords": ["home", "live", "nest", "habitat", "residence", "dwelling",
                             "casa", "viv", "ninho", "habitat", "residência", "morada"]
    },
    "What do you do in your daily life? What do you do during the day and at night? O que fazes no teu dia a dia? O que fazes durante o dia e à noite?": {
        "image": "stickers/routine.png",
        "caption": {
            "English": "🌙 Daily Life Detective!\nYou've discovered my secret schedule!",
            "Portuguese": "🌙 Detetive da Vida Diária!\nDescobriste o meu horário secreto!"
        },
        "semantic_keywords": ["daily", "routine", "day", "night", "schedule", "activities",
                             "diário", "rotina", "dia", "noite", "horário", "atividades"]
    },
    "What do you eat for food—and how do you catch it? O que comes — e como o apanhas?": {
        "image": "stickers/food.png",
        "caption": {
            "English": "🍽️ Food Finder!\nThanks for feeding your curiosity!",
            "Portuguese": "🍽️ Descobridor de Comida!\nObrigado por alimentares a tua curiosidade!"
        },
        "semantic_keywords": ["eat", "food", "diet", "prey", "hunt", "catch", "feed",
                             "comer", "comida", "dieta", "presa", "caçar", "apanhar", "alimentar"]
    },
    "How can I help you? What do you need from humans to help your species thrive? Como posso ajudar-te? O que precisas dos humanos para ajudar a tua espécie a prosperar?": {
        "image": "stickers/helper.png",
        "caption": {
            "English": "🌱 Species Supporter!\nYou care about our survival!",
            "Portuguese": "🌱 Apoiante de Espécies!\nTu importas-te com a nossa sobrevivência!"
        },
        "semantic_keywords": ["help", "support", "thrive", "survive", "conservation", "protect", "save",
                             "ajudar", "apoiar", "prosperar", "sobreviver", "conservação", "proteger", "salvar"]
    }
}

def semantic_match(user_input, question_key, reward_details):
    """
    Optimized semantic matching: Use invoke() instead of the deprecated __call__()
    """
    prompt = f"""
    Analyze whether the following two questions are similar in meaning:
    
    Original question: "{question_key}"
    User question: "{user_input}"
    
    Consider synonyms, paraphrasing, and different ways of asking the same thing.
    Also consider these relevant keywords: {reward_details.get('semantic_keywords', [])}
    
    Are these questions essentially asking the same thing? Respond only with 'yes' or 'no'.
    """
    
    # Optimization: Use invoke() instead of the deprecated __call__()
    response = semantic_model.invoke(prompt)
    return response.strip().lower() == 'yes'

def chat_message(name):
    if name == "assistant":
        return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(name=name, avatar="zino.png", width="content")
    else:
        return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(name=name, avatar=":material/face:", width="content")

# Language texts
language_texts = {
    "English": {
        "title": "Hi! I'm Fred,",
        "subtitle": "A Zino's Petrel.",
        "prompt": "What would you like to ask me?",
        "chat_placeholder": "Ask a question!",
        "tips_button": "Tips",
        "clear_button": "Clear and Restart",
        "friendship_score": "Friendship Score!",
        "score_description": "Unlock special stickers with your interactions",
        "doubtful": "Doubtful about the response?",
        "fact_check": "Fact-Check this answer",
        "fact_check_info": "Ask me a question to see the fact-check results based on scientific knowledge!",
        "loading_audio": "Preparing audio response...",
        "loading_thought": "Thinking about your question...",
        "gift_message": "After our wonderful conversation, I feel you deserve something special. \nPlease accept this medal as a symbol of your contribution to Madeira's biodiversity!",
        "medal_caption": "Biodiversity Trailblazer Medal",
        "sticker_toast": "You earned a new sticker!",
        "error_message": "I'm sorry, I had trouble processing that. Could you try again?",
        "voice_selector": "🎤 Voice",
        "loading_audio": "🎤 Voice Generating...",
        "voice_help": "Cherry: Female (lively) | Ethan: Male",
        "stickers_collected": "You've collected {current} out of {total} stickers!",
        "tips_content": """
        <div style="
            background-color: #fff;
            border: 2px solid #a1b065;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        ">
            <p style="margin-top: 0px;">Your <strong>Friendship Score</strong> grows based on how you talk to your critter friend. 🐦💚</p>
            <ul>
                <li>Ask about its habitat or life</li>
                <li>Show care or kindness</li>
                <li>Support nature and the planet</li>
                <li>Share your thoughts or feelings</li>
                <li>Be playful, curious, and respectful</li>
            </ul>
            <p style="margin-top: 10px;">💬 The more positive you are, the higher your score! 🌱✨ But watch out — unkind words or harmful ideas can lower your score. 🚫</p>
        </div>
        """,
        "tips_help": "Click to see tips on how to get a higher Friendship Score!",
        "clear_help": "Click to clear the chat history and start fresh!",
        "score_guide_title": "💡How the 'Friendship Score!' Works"
    },
    "Portuguese": {
        "title": "Olá! Eu sou o Fred,",
        "subtitle": "Uma Freira da Madeira.",
        "prompt": "O que gostarias de me perguntar?",
        "chat_placeholder": "Faz uma pergunta!",
        "tips_button": "Dicas",
        "clear_button": "Limpar e Recomeçar",
        "friendship_score": "Pontuação de Amizade!",
        "score_description": "Desbloqueia autocolantes especiais com as tuas interações",
        "doubtful": "Com dúvidas sobre a resposta?",
        "fact_check": "Verificar Factos desta resposta",
        "fact_check_info": "Faz-me uma pergunta para veres os resultados da verificação baseados em conhecimento científico!",
        "loading_audio": "A preparar resposta de áudio...",
        "loading_thought": "A pensar na tua pergunta...",
        "gift_message": "Após a nossa conversa maravilhosa, sinto que mereces algo especial. \nPor favor, aceita esta medalha como símbolo do teu contributo para a biodiversidade da Madeira!",
        "medal_caption": "Medalha de Pioneiro da Biodiversidade",
        "sticker_toast": "Ganhaste um autocolante novo!",
        "error_message": "Desculpa, tive problemas a processar isso. Podes tentar novamente?",
        "voice_selector": "🎤 Voz",
        "loading_audio": "🎤 A Gerar Voz...",
        "voice_help": "Cherry: Feminina (animada) | Ethan: Masculina",
        "stickers_collected": "Já colecionaste {current} de {total} autocolantes!",
        "tips_content": """
        <div style="
            background-color: #fff;
            border: 2px solid #a1b065;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        ">
            <p style="margin-top: 0px;">A tua <strong>Pontuação de Amizade</strong> cresce com base em como falas com o teu amigo animal. 🐦💚</p>
            <ul>
                <li>Pergunta sobre o habitat ou vida dele</li>
                <li>Mostra cuidado ou bondade</li>
                <li>Apoia a natureza e o planeta</li>
                <li>Partilha os teus pensamentos ou sentimentos</li>
                <li>Sê brincalhão, curioso e respeitoso</li>
            </ul>
            <p style="margin-top: 10px;">💬 Quanto mais positivo fores, maior será a tua pontuação! 🌱✨ Mas cuidado — palavras rudes ou ideias prejudiciais podem baixar a tua pontuação. 🚫</p>
        </div>
        """,
        "tips_help": "Clica para veres dicas sobre como obteres uma Pontuação de Amizade mais alta!",
        "clear_help": "Clica para limpar o histórico da conversa e começares de novo!",
        "score_guide_title": "💡Como Funciona a 'Pontuação de Amizade'!"
    }
}
# UI
def main():
    # Language state (initialize first)
    if "language" not in st.session_state:
        st.session_state.language = "English"  # Default language
        
    if 'tts_voice' not in st.session_state:
        st.session_state.tts_voice = 'Roy' 
        
    # Get current language texts
    texts = language_texts[st.session_state.language]
    
    # Other session state initialization
    if "has_interacted" not in st.session_state:
        st.session_state.has_interacted = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "show_score_guide" not in st.session_state:
        st.session_state.show_score_guide = False
    if "intimacy_score" not in st.session_state:
        st.session_state.intimacy_score = 0
    if 'gift_given' not in st.session_state:
        st.session_state.gift_given = False
    if "audio_played" not in st.session_state:
        st.session_state.audio_played = False
    if "awarded_stickers" not in st.session_state:
        st.session_state.awarded_stickers = []
    if "last_sticker" not in st.session_state:
        st.session_state.last_sticker = None
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "newly_awarded_sticker" not in st.session_state:
        st.session_state.newly_awarded_sticker = False
    if "gift_shown" not in st.session_state:
        st.session_state.gift_shown = False
    if "current_audio_html" not in st.session_state:
        st.session_state.current_audio_html = None
    if "audio_pending" not in st.session_state:
        st.session_state.audio_pending = False
    if "needs_background_processing" not in st.session_state:
        st.session_state.needs_background_processing = False
        
    st.set_page_config(layout="wide")

    st.markdown("""
        <style>
        .stApp {
            background: #cdd5ae;
        }
        
        /* Responsive Font Size */
        @media (max-width: 768px) {
            .responsive-title {
                font-size: 2rem !important;
            }
            .responsive-subtitle {
                font-size: 2rem !important;
            }
            .responsive-prompt {
                font-size: 1rem !important;
            }
        }
        
        @media (min-width: 769px) and (max-width: 1200px) {
            .responsive-title {
                font-size: 2.5rem !important;
            }
            .responsive-subtitle {
                font-size: 2.5rem !important;
            }
            .responsive-prompt {
                font-size: 1.125rem !important;
            }
        }
        
        @media (min-width: 1201px) {
            .responsive-title {
                font-size: 3rem !important;
            }
            .responsive-subtitle {
                font-size: 3rem !important;
            }
            .responsive-prompt {
                font-size: 1.25rem !important;
            }
        }

        /* Chat message container */
        .chat-message-container {
            display: flex;
            margin-bottom: 16px;
            max-width: 80%;
        }

        /* Chat input text and placeholder styling - WHITE TEXT */
        .stChatInput input::placeholder {
            color: #a1b065 !important;
            opacity: 1 !important;
            font-size: 16px;
        }

        .stChatInput textarea::placeholder {
            color: #888888 !important;
            opacity: 1 !important;
            font-size: 16px;
        }

        .stChatInput input {
            color: white !important;
            font-size: 16px;
            caret-color: #888888 !important;  /* White cursor */
        }

        .stChatInput textarea {
            color: #2d4f38!important;
            font-size: 16px;
            caret-color: #888888 !important;  /* White cursor */
        }
        
        /* Fixed chat input box background color - changed to white */
        .stChatInput > div {
            border-color: #345e42 !important;
            background-color: white !important;
            border-radius: 20px !important;
        }
        
        /* Input field inner background color */
        .stChatInput input, .stChatInput textarea {
            background-color: white !important;
            color: #2d4f38 !important;  
        }
        
        /* 输入框聚焦状态 */
        .stChatInput div[data-testid="stChatInput"]:focus-within {
            border-color: #a1b065 !important;
            box-shadow: 0 0 0 2px rgba(161, 176, 101, 0.3) !important;
        }
        
        /* User message container - align right */
        .user-container {
            margin-left: auto;
            justify-content: flex-end;
        }
        
        /* Assistant message container - align left */
        .assistant-container {
            margin-right: auto;
            justify-content: flex-start;
        }
        
        /* Message bubble styling */
        .message-bubble {
            padding: 12px 16px;
            border-radius: 16px;
            word-wrap: break-word;
        }
        
        /* User message styling */
        .user-bubble {
            background-color: #efe7e2;
            color: #2d4f38;
            border-radius: 16px 16px 0 16px;
            border-color: white !important;
            border-width: 2px;
        }
        
        /* Assistant message styling */
        .assistant-bubble {
            background-color: white;
            color: #2d4f38;
            border-radius: 16px 16px 16px 0;
        }
                
        .stChatMessage:has([data-testid="stChatMessageAvatarCustom"]) {
            display: flex;
            flex-direction: row-reverse;
            justify-self: end;
            background-color: white;
            color: black;
            border-radius: 16px 16px 0 16px;
            border-color: gray !important;
            border-width: 2px;
        }
        [data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
            text-align: right;
        }
                
        [class*="st-key-user"] {
            display: flex;
            align-items: flex-end;
            p {
                font-size: 1.125rem;
                color: black;
                font-weight: medium;
            }
                
        }
                
        .stChatMessage {
            background-color: transparent;
        }

        [class*="st-key-assistant"] {
            background-color: #345e42;
            border-radius: 16px 16px 16px 0;
            padding-right: 16px;
            border-color: white !important;
            border-width: 2px;
                
            p {
                font-size: 1.125rem;
                color: white;
                font-weight: medium;
                padding-left: 4px;
            }
                
            img {
                display: flex;
                height: 52px;
                width: 52px;
            }
        }
        
        .st-key-chat_section{
            display: flex;
            flex-direction: column-reverse;
            justify-content: flex-end;
        }
        /* Remove red border outline from chat input when active */
        .stChatInput div[data-testid="stChatInput"] > div:focus-within {
            box-shadow: none !important;
            border-color: #a1b065 !important;
            border-width: 1px !important;
        }
        
        /* Change chat input focus state */
        .stChatInput div[data-testid="stChatInput"]:focus-within {
            border-color: #a1b065 !important;
            box-shadow: 0 0 0 1px rgba(161, 176, 101, 0.5) !important;
        }
        
        /* Remove default Streamlit outlines */
        *:focus {
            outline: none !important;
        }
        
        /* Target specifically the chat input elements */
        [data-testid="stChatInput"] input:focus {
            box-shadow: none !important;
            outline: none !important;
            border-color: #a1b065 !important;
        }
        
        [data-testid="stChatInput"] textarea:focus {
            box-shadow: none !important;
            outline: none !important;
            border-color: #a1b065 !important;
        }
        button[kind="primary"] {
            background-color: #2b4e38;
            border: 0;
        }
        button[kind="primary"]:hover {
            background-color: #345e42;
            border: 0;
        }
        button[kind="secondary"] {
        
        }
        /* Style the selectbox options */
        .stSelectbox [data-testid="stMarkdownContainer"] p {
            color: white !important;
        }
        
        /* Style the selected value in the dropdown */
        .stSelectbox div[data-baseweb="select"] > div {
            color: white !important;
        }
        .sticker-reward {
            background-color: transparent;
            border: 2px solid #a1b065;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        .sticker-reward img {
            width: 200px;
        }
        .sticker-caption {
            font-size: 16px;
            margin-top: 8px;
            font-weight: bold;
        } 
        .gift-box {
            text-align: center;
            margin-top: 10px;
        }
        .gift-box img {
            width: 120px;
            margin-top: 10px;
        }  
        .friendship-score {
            margin-bottom: 32px;
            padding: 24px;
            border-radius: 16px;
        }
        .score-guide {
            position: fixed;
            bottom: 120px;
            left: calc(45% - 37%);
            width: 30%;
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            z-index: 101;
        }
        .close-btn {
            position: absolute;
            top: 5px;
            right: 5px;
            background: none;
            border: none;
            font-size: 16px;
            cursor: pointer;
        }
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            color: white;
        }
        .loading-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #a1b065;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>""", unsafe_allow_html=True)

    role = list(role_configs.keys())[0]
    role_config = role_configs[role]

    left_col, right_col = st.columns([0.63, 0.37], vertical_alignment="top", gap="large")
    
    # Store user input before processing
    user_input = None

    with left_col:
        with open("zino.png", "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0; padding: 0;">
                <div style="display: flex;">
                    <img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 200px;">
                </div>
                <div style="display: flex; flex-direction: column;">
                    <div>
                        <p style="margin-top: 0; font-weight: bold; font-size: 3rem; padding: 0; margin: 0;">{texts['title']}</p>
                        <p style="margin-top: 0; font-weight: bold; font-size: 3rem; padding: 0; margin: 0;">{texts['subtitle']}</p>
                    </div>
                    <p style="margin-bottom: 20px; font-weight: bold; padding: 0; font-size: 1.25rem;">{texts['prompt']}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Chat input (full width under title)
        user_input = st.chat_input(placeholder=texts['chat_placeholder'])
        
        print(f"User input: {user_input}")
        
        # Audio generation indicator (appears above chat history)
        audio_placeholder = st.empty()
        
        # Chat Section
        chatSection = st.container(key="chat_section", border=False)
        with chatSection:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            for message in st.session_state.chat_history:
                with chat_message(message["role"]):
                    # Check if this is a loading message
                    if message.get("is_loading", False):
                        st.markdown(f"""
                            <div class="loading-container">
                                <div class="loading-spinner"></div>
                                <div style="margin-left: 10px;">{message["content"]}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(message["content"])
    
        # Display persistent audio player if available
        if st.session_state.current_audio_html:
            components.html(st.session_state.current_audio_html, height=60)
    
    # Gift section (render in left column context)
        @st.dialog("🎁 Your Gift", width=680)
        def gift_dialog():
            with open("gift.png", "rb") as f:
                gift_img_base64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <div class="petrel-response gift-box">
                    <p>{texts['gift_message']}</p>
                    <img src="data:image/png;base64,{gift_img_base64}">
                    <div class="sticker-caption">{texts['medal_caption']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        if st.session_state.gift_given and not st.session_state.gift_shown: 
            gift_dialog()
            st.session_state.gift_shown = True
        
        # Generate audio after answer is displayed
        if st.session_state.audio_pending and st.session_state.last_answer:
            # Show audio generation indicator above chat history
            audio_placeholder.markdown(f"""
                <div class="loading-container" style="justify-content: flex-start; margin-top: 10px;">
                    <div class="loading-spinner"></div>
                    <div style="margin-left: 10px; color: #000000;">{texts['loading_audio']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                current_language = st.session_state.get('language', 'English')
                voice = st.session_state.get('tts_voice', 'Cherry')
                
                success, result, method = tts_speak(
                    st.session_state.last_answer, 
                    voice=voice, 
                    timeout=10,
                    language=current_language,
                    portuguese_variant="european"
                )
                
                if success:
                    # Store audio HTML in session state for persistent display
                    st.session_state.current_audio_html = result
                    print(f"[TTS] ✅ Audio generated using {method} for {current_language}")
                else:
                    st.session_state.current_audio_html = None
                    print(f"[TTS] ❌ {result}")
            except Exception as tts_error:
                st.session_state.current_audio_html = None
                print(f"[TTS] ❌ Exception: {tts_error}")
            
            # Clear the loading indicator
            audio_placeholder.empty()
            
            st.session_state.audio_played = True
            st.session_state.audio_pending = False
            st.rerun()
        
        # Background processing - run scoring and sticker checks AFTER answer is displayed
        if st.session_state.needs_background_processing and st.session_state.last_question:
            current_input = st.session_state.last_question
            
            # Update intimacy score (LLM call)
            update_intimacy_score(current_input)
            gift_triggered = check_gift()
            
            # Check for sticker rewards
            normalized_input = current_input.strip().lower()
            if not hasattr(st.session_state, 'last_processed_for_sticker') or st.session_state.last_processed_for_sticker != current_input:
                st.session_state.newly_awarded_sticker = False
                
                for q, reward in sticker_rewards.items():
                    exact = q.lower() == normalized_input
                    keywords = reward.get('semantic_keywords', [])
                    keyword_matches = sum(1 for keyword in keywords if keyword.lower() in normalized_input)
                    keyword_match = keyword_matches >= 2
                    
                    # Only call semantic_match if keyword matching didn't work
                    is_semantic_match = False
                    if not exact and not keyword_match:
                        is_semantic_match = semantic_match(normalized_input, q, reward)
                                        
                    if exact or keyword_match or is_semantic_match:
                        sticker_key = reward["image"]
                        if sticker_key not in [s["key"] for s in st.session_state.awarded_stickers]:
                            caption = reward["caption"][st.session_state.language] if isinstance(reward["caption"], dict) else reward["caption"]
                            st.session_state.awarded_stickers.append({
                                "key": sticker_key,
                                "image": reward["image"],
                                "caption": caption
                            })
                            st.session_state.newly_awarded_sticker = True
                            print(f"✨ Sticker awarded: {sticker_key}")
                        break
                
                st.session_state.last_processed_for_sticker = current_input
            
            # Mark background processing as complete
            st.session_state.needs_background_processing = False
            
            # Only rerun if a sticker was awarded (to show the toast notification)
            if st.session_state.newly_awarded_sticker:
                st.rerun()
            
        

    with right_col:
        # Language switcher
        # st.markdown("**Language / Idioma:**")
        # col1, col2 = st.columns(2)
        # with col1:
        #     if st.button("🇬🇧 English", use_container_width=True, 
        #                 type="primary" if st.session_state.language == "English" else "secondary"):
        #         st.session_state.language = "English"
        #         st.rerun()
        # with col2:
        #     if st.button("🇵🇹 Português", use_container_width=True,
        #                 type="primary" if st.session_state.language == "Portuguese" else "secondary"):
        #         st.session_state.language = "Portuguese"
        #         st.rerun()
        
        # Define dialog function outside of button conditional
        @st.dialog(texts['score_guide_title'], width="large") 
        def score_guide():
            st.markdown(texts['tips_content'], unsafe_allow_html=True)
        
        # Tips and Clear buttons
        input_section_col1, input_section_col2 = st.columns([0.35, 0.65], gap="small")
        with input_section_col1:
            if st.button(texts['tips_button'], icon=":material/lightbulb:", 
                        help=texts['tips_help'], 
                        use_container_width=True, type="primary"):
                score_guide()
                
        with input_section_col2:
            if st.button(texts['clear_button'], icon=":material/chat_add_on:", 
                        help=texts['clear_help'],  
                        use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.show_score_guide = False
                st.session_state.audio_played = True
                st.session_state.gift_given = False
                st.session_state.intimacy_score = 0
                st.session_state.awarded_stickers = []
                st.session_state.last_question = ""
                st.session_state.has_interacted = False
                st.session_state.processing = False
                st.session_state.most_relevant_texts = []
                st.session_state.last_answer = ""
                st.session_state.last_sticker = None
                st.session_state.last_analysis = {}
                st.session_state.newly_awarded_sticker = False
                st.session_state.gift_shown = False
                st.session_state.current_audio_html = None
                st.session_state.fact_check_cache = {}
                if "session_id" in st.session_state:
                    del st.session_state["session_id"]
                if "logged_interactions" in st.session_state:
                    del st.session_state["logged_interactions"]
                st.rerun()
        
        # Friendship score section
        current_score = min(6, int(round(st.session_state.intimacy_score)))
        
        st.markdown(f"""
        <div class="friendship-score">
            <div style="font-size:18px; font-style: italic; font-weight:bold; color:#31333e; text-align: left;">
                {texts['friendship_score']}
            </div>
            <div style="font-size:16px; color:#31333e; text-align: left;">{texts['score_description']}</div>
            <div style="font-size:24px; margin:5px 0; text-align: left;">
                <span style="color:#ff6b6b;">{'❤️' * current_score}</span>
                <span style="color:#ddd;">{'🤍' * (6 - current_score)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the most recent sticker if any exist
        if st.session_state.awarded_stickers:
            # Get the most recent sticker (last in the list)
            most_recent = st.session_state.awarded_stickers[-1]

            current_caption = most_recent["caption"]
            for q, reward in sticker_rewards.items():
                if reward["image"] == most_recent["image"]:
                    if isinstance(reward["caption"], dict) and st.session_state.language in reward["caption"]:
                        current_caption = reward["caption"][st.session_state.language]
                    break

            st.markdown(
                f"""
                <div class="sticker-reward">
                    <img src="data:image/png;base64,{base64.b64encode(open(most_recent["image"], "rb").read()).decode()}">
                    <div class="sticker-caption">{current_caption}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Add a small indicator showing how many stickers have been collected
            total_possible = len(sticker_rewards)
            total_collected = len(st.session_state.awarded_stickers)
            
            st.markdown(
                f"""
                <div style="text-align: center; font-size: 14px; margin-top: -10px; color: #555; margin-bottom: 20px;">
                    {texts['stickers_collected'].format(current=total_collected, total=total_possible)}
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Fact Check Section
        st.markdown(f"""
            <div style="font-size:18px; font-style: italic; font-weight:bold; color:#31333e; text-align: left;">
                {texts['doubtful']}
            </div>
        """, unsafe_allow_html=True)
        
        # Generate fact-check content only once and cache it
        if "most_relevant_texts" in st.session_state and "last_question" in st.session_state and "last_answer" in st.session_state:
            # Create a unique key for this Q&A pair
            fact_check_key = f"{st.session_state.last_question}_{st.session_state.last_answer}"
            
            # Only generate if not already cached for this Q&A
            if "fact_check_cache" not in st.session_state:
                st.session_state.fact_check_cache = {}
            
            if fact_check_key not in st.session_state.fact_check_cache: 
                if len(st.session_state.most_relevant_texts) > 0:
                    try:
                        fact_check_summary = generate_fact_check_content(
                            question=st.session_state.last_question,
                            retrieved_docs=st.session_state.most_relevant_texts,
                            ai_answer=st.session_state.last_answer,
                            language=st.session_state.language
                        )
                        st.session_state.fact_check_cache[fact_check_key] = fact_check_summary
                    except Exception as e:
                        print(f"[Fact-Check] Abstract generation failed: {str(e)}")
                        st.session_state.fact_check_cache[fact_check_key] = st.session_state.most_relevant_texts[0].page_content[:300] + "..."
        
        # Display the expander with cached content
        with st.expander(texts['fact_check'], expanded=False):
            if "most_relevant_texts" in st.session_state and "last_question" in st.session_state and "last_answer" in st.session_state:
                fact_check_key = f"{st.session_state.last_question}_{st.session_state.last_answer}"
                
                if fact_check_key in st.session_state.fact_check_cache:
                    st.markdown("""
                        <style>
                        .fact-check-box {
                            background: #f0f8ff;
                            padding: 20px;
                            border-radius: 10px;
                            margin: 10px 0;
                            border-left: 4px solid #4a90e2;
                            color: #2c3e50;
                            line-height: 1.6;
                        }
                        .fact-check-box p {
                            margin-bottom: 10px;
                        }
                        .fact-check-box strong {
                            color: #1e3a8a;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(st.session_state.fact_check_cache[fact_check_key])
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Generating fact-check...")
            else:
                st.info(texts['fact_check_info'])
    
    # Process user input AFTER both columns are fully rendered
    if user_input and user_input != st.session_state.last_question and not st.session_state.processing:
        # Add user message to chat history immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.last_question = user_input
        
        # Add temporary loading message with special marker
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": texts['loading_thought'],
            "is_loading": True
        })
        
        # Set processing state and trigger immediate rerun
        st.session_state.processing = True
        st.session_state.has_interacted = True
        st.session_state.show_score_guide = False
        
        # Trigger rerun
        st.rerun()
    
    # Process AI response when we're in processing state (happens after rerun)
    if st.session_state.processing and st.session_state.last_question:
        try:
            current_input = st.session_state.last_question
            
            # Remove the loading message from chat history
            if st.session_state.chat_history and st.session_state.chat_history[-1].get("is_loading", False):
                st.session_state.chat_history.pop()
            
            # Process response
            try:
                # Using an Optimized RAG Retriever (with Caching)
                rag = get_rag_instance(
                    persist_directory=get_vectordb(role),
                    dashscope_api_key=dashscope_key
                )

                # Decide retrieval size (still useful)
                k_value = 4  # keep stable for museum kiosk; 3–5 is good

                most_relevant_texts = rag.retrieve(
                    query=current_input,
                    k=k_value,
                    lambda_mult=0.3,
                    relevance_threshold=None
                )
                print(f"Retrieved {len(most_relevant_texts)} relevant documents for the query.")

                # Always use English chain/system prompt
                chain, role_config = get_conversational_chain(role, "English")

                raw_answer = chain.invoke({"input_documents": most_relevant_texts, "question": current_input})

                # Handling the dictionary format returned by invoke()
                answer_text = raw_answer.get("output_text", raw_answer) if isinstance(raw_answer, dict) else raw_answer
                answer = re.sub(r'^\s*Answer:\s*', '', answer_text).strip()
                st.session_state.last_answer = answer

                # Save results to session state
                st.session_state.most_relevant_texts = most_relevant_texts
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Mark that we need to process scoring/stickers in background
                st.session_state.needs_background_processing = True
                
                # Set up for audio generation on next render
                st.session_state.audio_pending = True
                st.session_state.processing = False
                
                # Trigger rerun to display the new message immediately
                st.rerun()
                
            except Exception as e:
                # Handle processing errors
                print(f"Error processing response: {str(e)}")
                    
                error_msg = texts['error_message']
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.session_state.processing = False
                st.rerun()
        
        except Exception as outer_e:
            # Handle any unexpected errors
            print(f"Outer exception in processing handling: {str(outer_e)}")
            st.session_state.processing = False
            # Remove loading message if present
            if st.session_state.chat_history and st.session_state.chat_history[-1].get("is_loading", False):
                st.session_state.chat_history.pop()
            st.rerun()
    
    cleanup_audio_files()

    # Log the interaction to Supabase
    if st.session_state.last_question:
        # Check if this specific interaction has already been logged
        if "logged_interactions" not in st.session_state:
            st.session_state.logged_interactions = set()
        
        combined = f"{st.session_state.last_question}|{st.session_state.last_answer}"

        interaction_key = hashlib.md5(combined.encode()).hexdigest()
        if interaction_key not in st.session_state.logged_interactions:
            log_interaction(
                user_input=st.session_state.last_question,
                ai_response=st.session_state.last_answer,
                intimacy_score=st.session_state.intimacy_score,
                is_sticker_awarded=st.session_state.newly_awarded_sticker,
                gift_given=st.session_state.gift_given
            )
            st.session_state.logged_interactions.add(interaction_key)

if __name__ == "__main__":
    main()
