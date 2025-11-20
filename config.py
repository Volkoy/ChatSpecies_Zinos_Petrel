"""
Configuration Management Module
Load all configuration items from .env files
"""

from dotenv import load_dotenv
import os

# Load Environment Variables
load_dotenv()

class Config:
    """Application Configuration Class"""
    
    # ==================== Qwen LLM Configuration ====================
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    QWEN_MODEL = os.getenv("QWEN_MODEL_NAME", "qwen-turbo")
    
     # Temperature Parameters
    TEMP_CONVERSATION = float(os.getenv("QWEN_TEMPERATURE_CONVERSATION", "0.0"))
    TEMP_SCORING_POS = float(os.getenv("QWEN_TEMPERATURE_SCORING_POS", "0.2"))
    TEMP_SCORING_NEG = float(os.getenv("QWEN_TEMPERATURE_SCORING_NEG", "0.0"))
    TEMP_SEMANTIC = float(os.getenv("QWEN_TEMPERATURE_SEMANTIC", "0.4"))
    TEMP_ROUTER = float(os.getenv("QWEN_TEMPERATURE_ROUTER", "0.0"))
    
    # ==================== Vector Embedding Configuration ====================
    QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v2")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "db5")
    
    # ==================== Qwen TTS Configuration ====================
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "qwen")
    QWEN_TTS_MODEL = os.getenv("QWEN_TTS_MODEL", "qwen3-tts-flash")
    QWEN_TTS_VOICE = os.getenv("QWEN_TTS_VOICE", "Cherry")
    QWEN_TTS_LANGUAGE = os.getenv("QWEN_TTS_LANGUAGE", "Chinese")
    QWEN_TTS_STREAM = os.getenv("QWEN_TTS_STREAM", "true").lower() == "true"
    USE_GTTS_FALLBACK = os.getenv("USE_GTTS_FALLBACK", "true").lower() == "true"
    
    # ==================== RAG Retrieval Configuration ====================
    RAG_MMR_K = int(os.getenv("RAG_MMR_K", "4"))
    RAG_MMR_FETCH_K = int(os.getenv("RAG_MMR_FETCH_K", "20"))
    RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))
    ENABLE_HISTORY_DEDUP = os.getenv("ENABLE_HISTORY_DEDUP", "true").lower() == "true"
    MAX_HISTORY_ROUNDS = int(os.getenv("MAX_HISTORY_ROUNDS", "10"))
    
    # Hybrid Search (Optional)
    ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true"
    HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
    HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))
    
    # Reorder (Optional)
    ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
    RERANKING_MODEL = os.getenv("RERANKING_MODEL", "cohere")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
    COHERE_RERANK_TOP_N = int(os.getenv("COHERE_RERANK_TOP_N", "3"))
    
    # ==================== Database Configuration ====================
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_TABLE_NAME = os.getenv("SUPABASE_TABLE_NAME", "interactions")
    
    # ==================== Agent Configuration ====================
    USE_WEB_SEARCH = os.getenv("USE_WEB_SEARCH", "true").lower() == "true"
    WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")
    ENABLE_SMART_ROUTING = os.getenv("ENABLE_SMART_ROUTING", "true").lower() == "true"
    ROUTING_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTING_CONFIDENCE_THRESHOLD", "0.7"))
    
    # Tavily（Optional）
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
    
    # ==================== Application Configuration ====================
    APP_NAME = os.getenv("APP_NAME", "Monk Seal")
    APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
    APP_DEBUG = os.getenv("APP_DEBUG", "false").lower() == "true"
    
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))
    MAX_INTIMACY_SCORE = int(os.getenv("MAX_INTIMACY_SCORE", "6"))
    
    # ==================== Function Switch ====================
    FEATURE_QWEN_TTS = os.getenv("FEATURE_QWEN_TTS", "true").lower() == "true"
    FEATURE_SMART_AGENT = os.getenv("FEATURE_SMART_AGENT", "true").lower() == "true"
    FEATURE_HYBRID_RAG = os.getenv("FEATURE_HYBRID_RAG", "false").lower() == "true"
    FEATURE_GIFT_SYSTEM = os.getenv("FEATURE_GIFT_SYSTEM", "true").lower() == "true"
    FEATURE_VOICE_SELECTION = os.getenv("FEATURE_VOICE_SELECTION", "true").lower() == "true"
    
    # ==================== Backup Configuration ====================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ENABLE_OPENAI_FALLBACK = os.getenv("ENABLE_OPENAI_FALLBACK", "false").lower() == "true"
    FALLBACK_VECTOR_DB_PATH = os.getenv("FALLBACK_VECTOR_DB_PATH", "db5")
    
    # ==================== Log Configuration ====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    ENABLE_API_LOGGING = os.getenv("ENABLE_API_LOGGING", "true").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Verification requires configuration"""
        required_configs = {
            'DASHSCOPE_API_KEY': cls.DASHSCOPE_API_KEY,
            'SUPABASE_URL': cls.SUPABASE_URL,
            'SUPABASE_KEY': cls.SUPABASE_KEY,
        }
        
        missing = [key for key, value in required_configs.items() if not value]
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (for debugging)）"""
        print("=" * 50)
        print("Current Configuration")
        print("=" * 50)
        print(f"Qwen Model: {cls.QWEN_MODEL}")
        print(f"TTS Provider: {cls.TTS_PROVIDER}")
        print(f"TTS Voice: {cls.QWEN_TTS_VOICE}")
        print(f"Vector DB: {cls.VECTOR_DB_PATH}")
        print(f"Smart Agent: {cls.FEATURE_SMART_AGENT}")
        print(f"Voice Selection: {cls.FEATURE_VOICE_SELECTION}")
        print("=" * 50)

# Create a global configuration instance
config = Config()

# Automatic verification (optional; comment out to skip)
# config.validate()

