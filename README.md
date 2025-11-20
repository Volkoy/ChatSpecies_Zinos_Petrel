# 🐦 ChatSpecies - AI Interactive Education System

<div align="center">

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Qwen](https://img.shields.io/badge/Qwen-4A90E2?style=for-the-badge&logo=ai&logoColor=white)](https://tongyi.aliyun.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

**Chat with the special species in Madeira and explore ecological conservation knowledge**

[Quick Start](#-quick-start) • [Features](#-features) • [Online Experience](#-online-experience) • [Deployment Guide](#-deployment-guide) • [Documentation](#-documentation)

</div>

---

## 📖 Project Introduction

ChatSpecies is an innovative AI interactive education system that allows users to converse with the special species in Madeira. The system combines:

- 🤖 **Qwen AI Model** - Intelligent dialogue and speech synthesis
- 📚 **RAG Knowledge Base** - Authoritative knowledge based on 1298 scientific document chunks
- 🔍 **Smart Search** - Automatic web search filtering irrelevant content
- 🎁 **Interactive System** - Friendship score rating and sticker rewards
- 🌐 **Bilingual Support** - English & Português

---

## ✨ Features

### 🗣️ Intelligent Dialogue System
- **Role-playing**: Creatures persona for an authentic interactive experience
- **Speech Synthesis**: Qwen TTS provides natural and fluent speech in English and Azure TTS for Portuguese
- **Bilingual Switching**: Seamless switching between English/Portuguese

### 📚 RAG Knowledge Enhancement
- **Authoritative Knowledge Base**: Based on 18 scientific papers (1298 document chunks)
- **Intelligent Retrieval**: MMR algorithm ensures diversity and relevance
- **Vector Database**: ChromaDB + Qwen Embeddings (text-embedding-v3)

### 🔍 Intelligent Fact Verification
- **AI Summary Generation**: Automatically summarizes knowledge base content
- **Smart Web Search**:
  - Optimizes search queries based on RAG context
  - Automatically filters irrelevant content (programming frameworks, technical docs, etc.)
  - DuckDuckGo free search (no limits)
- **Source Attribution**: Automatically cites literature and page numbers

### 🎁 Interactive Incentive System
- **❤️ Friendship Score**: Intelligent scoring based on dialogue quality
- **🎁 Sticker Rewards**: Unlock special themed stickers (Food, Help, Home, Daily)
- **🏅 Achievement Badges**: Receive a mysterious gift upon reaching a perfect score

---

## 🚀 Quick Start

### Method 1: Local Deployment (Recommended) ⭐

#### 1. Clone the Project
```bash
git clone https://github.com/your-username/zinos-chat.git
cd zinos-chat
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Configure API Keys
Copy config.env.template to .env and fill in your API Keys:

```env
# Required Configuration
DASHSCOPE_API_KEY=sk-your-qwen-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key

# Optional Configuration
USE_WEB_SEARCH=true
WEB_SEARCH_PROVIDER=duckduckgo
OPENAI_API_KEY="your openai api key"
ENABLE_OPENAI_FALLBACK=false
AZURE_TTS_KEY="your azure tts key"
AZURE_TTS_REGION=westeurope
```

**Get API Keys:**
- [Qwen API](https://dashscope.aliyun.com/) - Free tier available
- [Supabase](https://supabase.com/) - Free plan sufficient
- [Microsoft Azure](https://azure.microsoft.com/en-us/products/ai-foundry/tools/speech) - Free tier available

#### 4. Set up RAG Knowledge Base
```bash
# Windows
setup_rag_system.bat

# Mac/Linux
pip install tqdm
python vectorize_knowledge_base.py
```

**Expected Output：**
```
✅ Vector database created successfully!
📊 Statistics:
   - Document Count: 1298 blocks
   - Embedding Model: text-embedding-v3
   - Vector Store Path: db5_qwen
```

#### 5. nable Smart Web Search (Optional)
```bash
# Already included in requirements.txt, no extra installation needed
# For separate installation:
pip install ddgs
```

#### 6. Run the Application
```bash
streamlit run main.py
```

Visit: http://localhost:8501

---

### Method 2: Online Experience 🌐

#### Streamlit Cloud Deployment

1. Fork this project to your GitHub
2. Visit [Streamlit Cloud](https://share.streamlit.io/)
3. Connect the GitHub repository and deploy
4. Configure Secrets in Streamlit settings:
```toml
DASHSCOPE_API_KEY = "sk--your-key"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-key"
USE_WEB_SEARCH = "true"
WEB_SEARCH_PROVIDER = "duckduckgo"
OPENAI_API_KEY="your openai api key"
ENABLE_OPENAI_FALLBACK=false
AZURE_TTS_KEY="your azure tts key"
AZURE_TTS_REGION="westeurope"
```

**Detailed Steps**: Refer to [QUICK_DEPLOY.md](QUICK_DEPLOY.md)

---

## 📊 Project Structure

```
ChatSpeciest/
├── main.py                          # Main Application
├── config.py                        # Configuration Management
├── requirements.txt                 # Dependencies List
├── config.env.template              # Configuration Template
│
├── Core Modules/
│   ├── rag_utils.py                 # RAG Retrieval Logic
│   ├── fact_check_utils.py          # Fact Verification (Summary+Search)
│   └── tts_utils.py                 # Speech Synthesis
│
├── Knowledge Base/
│   ├── Zino's Petrel/               # 18 PDF Scientific Papers
│   ├── vectorize_knowledge_base.py  # Vectorization Script
│   └── db5_qwen/                    # Vector Database (Auto-generated)
│
├── Test Scripts/
│   ├── test_rag_quality.py          # RAG Quality Test
│   └── test_user_questions.py       # User Questions Test
│
├── Tool Scripts/
│   └── setup_rag_system.bat         # RAG One-Click Setup
│
├── Resource Files/
│   ├── zino.png                     # Zino Avatar
│   ├── gift.png                     # Gift Image
│   └── stickers/                    # Sticker Images
│
└── Documentation/
    ├── README.md                    # Project Main Doc (This File)
    └── QUICK_DEPLOY.md              # Quick Deployment Guide
```

---

## 🧪 Testing & Validation

### RAG Quality Test
```bash
# Full Test
python test_rag_quality.py

# User Questions Test
python test_user_questions.py
```

**Expected Results：**
- ✅ Retrieval Quality: Excellent (Coverage ≥75%)
- ✅ Pass Rate: ≥78%
- ✅ Average Coverage: ≥50%

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|------|------|------|
| **AI Model** | Qwen (Tongyi Qianwen)、OpenAI、Microsoft Azure| LLM、Embeddings、TTS |
| **Frontend Framework** | Streamlit | Web App Interface |
| **Vector Database** | ChromaDB | Knowledge Base Storage |
| **RAG Framework** | LangChain | Retrieval-Augmented Generation |
| **Web Search** | DuckDuckGo (ddgs) | Free Internet Search |
| **Database** | Supabase | Interaction Log Storage |
| **Document Processing** | PyPDF | PDF Parsing |

---

## 📈 Performance Metrics

### RAG Retrieval Quality
- **Document Count**: 1298 blocks
- **Retrieval Precision**: ~90%（Keyword Coverage）
- **Average Response Time**: <1 second

### Smart Search Optimization
- **Search Accuracy**: ~90%（After Optimization）
- **Relevant Results Ratio**: 100%（After Filtering）
- **Irrelevant Results Count**: 0（Automatically Filtered）

### User Experience
- **Dialogue Fluency**: ⭐⭐⭐⭐⭐
- **Knowledge Accuracy**: ⭐⭐⭐⭐⭐
- **Interface Friendliness**: ⭐⭐⭐⭐⭐

---

## 📚 Documentation

| Document | Description |
|------|------|
| [QUICK_DEPLOY.md](QUICK_DEPLOY.md) | Quick Deployment Guide（⭐ Recommended for Beginners） |
| [docs/COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md) | Complete Usage Guide |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common Issue Resolution |
| [docs/RAG_SETUP_GUIDE.md](docs/RAG_SETUP_GUIDE.md) | RAG System Setup |
| [docs/WEB_SEARCH_GUIDE.md](docs/WEB_SEARCH_GUIDE.md) | Web Search Configuration |
| [docs/SMART_SEARCH_QUICK_START.md](docs/SMART_SEARCH_QUICK_START.md) | Smart Search Optimization |
| [docs/TEST_GUIDE.md](docs/TEST_GUIDE.md) | Testing Guide |

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><b>1. DDGS Package Error</b></summary>

**Error：** `DDGS.text() missing 1 required positional argument: 'query'`

**Solution：**
```bash
# Uninstall old package, install new package
pip uninstall duckduckgo-search -y
pip install ddgs
```
</details>

<details>
<summary><b>2. Empty Vector Database</b></summary>

**Error：** `Document Count: 0`

**Solution：**
```bash
.\fix_vectordb.bat

# Or manual fix
pip install ddgs
python vectorize_knowledge_base.py
```
</details>

<details>
<summary><b>3. TTS Speech Failure</b></summary>

**Error：** `Qwen TTS failed`

**Solution：** Check configuration in .env
```env
QWEN_TTS_MODEL=qwen3-tts-flash
QWEN_TTS_VOICE=Cherry
```
</details>

**More Issues**: Refer to [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- **Qwen OpenAI Microsoft Azure** - Providing powerful AI capabilities
- **Streamlit** - Excellent web framework
- **LangChain** - RAG framework support
- **Scientific Literature Contributors** - Providing Species research materials

---

<div align="center">

**Using AI to protect endangered species, making education warmer** 💙

[⬆ Back to top](#-ChatSpecies---ai-interative educational system)

</div>
