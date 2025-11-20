# 🚀 ChatSpecies - Quick Deployment Guide

**3 Deployment Methods, Go Live in 10 Minutes!**

---

## 📋 Pre-deployment Preparation

### Required API Keys

| API | Purpose | Get Address | Cost |
|-----|---------|-------------|------|
| **Qwen API** | LLM + TTS + Embeddings | [DashScope](https://dashscope.aliyun.com/) | Free tier available |
| **Supabase** | Interaction Log Storage | [Supabase](https://supabase.com/) | Free plan sufficient |

### Optional API Keys

| API | Purpose | Get Address | Cost |
|-----|---------|-------------|------|
| **Tavily** | High-quality Web Search | [Tavily](https://tavily.com/) | 1000 free searches/month |

---

## 🎯 Method 1: Local Deployment (Windows) ⭐ Recommended for Beginners

### Step 1: Clone the Project
```bash
git clone https://github.com/your-username/zinos-chat.git
cd zinos-chat
```

### Step 2: Install Dependencies
```bash
# Using pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
```bash
# 1. Copy configuration template
copy config.env.template .env

# 2. Edit .env file, fill in your API Keys
notepad .env
```

**Required Configuration：**
```env
DASHSCOPE_API_KEY=sk-your-qwen-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key
```

**Optional Configuration：**
```env
USE_WEB_SEARCH=true
WEB_SEARCH_PROVIDER=duckduckgo
TAVILY_API_KEY=tvly-your-key
OPENAI_API_KEY="your openai api key"
ENABLE_OPENAI_FALLBACK=false
AZURE_TTS_KEY="your azure tts key"
AZURE_TTS_REGION=westeurope
```

### Step 4: Set up RAG Knowledge Base
```bash
# One-click setup (Recommended)
setup_rag_system.bat

# Or execute manually
pip install tqdm
python vectorize_knowledge_base.py
```

**Wait 5-10 minute**，after completion you should see：
```
✅ 向Vector database created successfully!
📊 Statistics:
   - Document Count: 1298 blocks
   - Embedding Model: text-embedding-v3
```

### Enable Smart Web Search (Optional)
```bash
# Already included in requirements.txt, no additional action needed
# Web search functionality will be automatically enabled
```

### Run the Application
```bash
streamlit run main.py
```

**Visit**: http://localhost:8501

---

## 🌐 Method 2: Streamlit Cloud Deployment (Online Access)）

### Step 1: Prepare GitHub Repository
```bash
# 1. Fork or push the project to your GitHub
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Visit [Streamlit Cloud](https://share.streamlit.io/)
2. Click "New app"
3. Select your GitHub repository
4. Configure：
   - **Main file path**: `main.py`
   - **Python version**: 3.11

### Step 3: Configure Secrets

On the Streamlit Cloud settings page, add the following Secrets:

```toml
# .streamlit/secrets.toml

# Required Configuration
DASHSCOPE_API_KEY = "sk-your-qwen-key"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-supabase-key"

# Optional Configuration
USE_WEB_SEARCH = "true"
WEB_SEARCH_PROVIDER = "duckduckgo"
TAVILY_API_KEY = "tvly-your-key"
OPENAI_API_KEY="your openai api key"
ENABLE_OPENAI_FALLBACK=false
AZURE_TTS_KEY="your azure tts key"
AZURE_TTS_REGION="westeurope"

# Model Configuration
QWEN_MODEL_NAME = "qwen-turbo"
QWEN_EMBEDDING_MODEL = "text-embedding-v3"
QWEN_TTS_MODEL = "qwen3-tts-flash"
QWEN_TTS_VOICE = "Cherry"
```

### Step 4: Deploy and Test

1. Click "Deploy"
2. Wait for deployment to complete (approx. 2-3 minutes)
3. Visit your application link

---

## 🐧 Method 3: Linux/Mac Deployment

### Step 1: Clone the Project
```bash
git clone https://github.com/your-username/monkseal-chat.git
cd zinos-chat
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Python venv
python3 -m venv venv
source venv/bin/activate

# Or use conda
conda create -n zinos python=3.11
conda activate zinos
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```bash
# Copy configuration template
cp config.env.template .env

# Edit configuration
nano .env
# or vim .env
```

### Step 5: Set up RAG Knowledge Base
```bash
pip install tqdm
python vectorize_knowledge_base.py
```

### Step 6: Enable Web Search (Optional)
```bash
pip install ddgs
```

Add to `.env` ：
```env
USE_WEB_SEARCH=true
WEB_SEARCH_PROVIDER=duckduckgo
```

### Step 7: Run the Application
```bash
streamlit run main.py
```

---

## 🧪 Post-deployment Testing

### 1. Basic Functionality Test

After accessing the application:
1. ✅ Select language（English/Português）
2. ✅ Enter question："Hi, how are you?"
3. ✅ Check AI response
4. ✅ Check voice playback


### 2. RAG Quality Test
```bash
# Full test
python test_rag_quality.py

# Quick test
python test_user_questions.py
```

**Expected Results：**
```
✅ Vector Store Path: db5_qwen
✅ Document Count: 1298
✅ Retrieval Quality: Excellent (Coverage ≥75%)
```

### 3. Web Search Test
```bash
python test_smart_search.py
```

**Expected Result：**
```
✅ Search query optimization normal
✅ Result filtering normal (no technical/programming content)
✅ All tests passed
```

---

## 🔧 Common Deployment Issues

### Issue 1: DDGS Package Error

**Error:**
```
DDGS.text() missing 1 required positional argument: 'query'
```

**Solution：**
```bash
# Uninstall old package, install new package
pip uninstall duckduckgo-search -y
pip install ddgs
```

---

### Issue 2: Empty Vector Database

**Error：**
```
Document Count: 0
```

**Solution：**
```bash
# Ensure embedding model configuration is correct
# In .env:
QWEN_EMBEDDING_MODEL=text-embedding-v3

# Re-vectorize
python vectorize_knowledge_base.py
```

---

### Issue 3: Streamlit Cloud Deployment Failed

**Error：**
```
ModuleNotFoundError: No module named 'ddgs'
```

**Solution：**
Ensure `requirements.txt` contains：
```
ddgs
tavily-python
```

---

### Issue 4: Supabase Connection Failed

**Error：**
```
Connection to Supabase failed
```

**Solution：**
1. Check if Supabase URL and Key are correct
2. Ensure database tables are created:
   - Run `create_table_interactions.sql`
   - Or create manually in Supabase Dashboard

---

## 📊 Deployment Checklist

### Environment Configuration ✅

- [ ] Python 3.11+ installed
- [ ] All dependencies installed（`pip install -r requirements.txt`）
- [ ] `.env` file configured
- [ ] Qwen API Key valid
- [ ] Supabase URL and Key valid

### RAG System ✅

- [ ] Vector database created（`db5_qwen/`）
- [ ] Document Count = 1298
- [ ] Embedding Model = text-embedding-v3
- [ ] RAG test passed（`test_rag_quality.py`）

### Web Search  ✅

- [ ] DDGS package correctly installed
- [ ] `USE_WEB_SEARCH=true` configured
- [ ] Search test passed（`test_smart_search.py`）

### Application Functionality ✅

- [ ] Application accessible normally
- [ ] Dialogue functionality normal
- [ ] Speech synthesis normal
- [ ] Fact-Check functionality normal
- [ ] Bilingual switching normal

---

## 🚀 Quick Command Reference

### Windows
```bash
# Complete deployment process
git clone <repo>
cd zinos-chat
pip install -r requirements.txt
copy config.env.template .env
# Edit .env to fill in API Keys
setup_rag_system.bat
streamlit run main.py
```

### Linux/Mac
```bash
# Complete deployment process
git clone <repo>
cd zinos-chat
pip install -r requirements.txt
cp config.env.template .env
# Edit .env to fill in API Keys
pip install tqdm ddgs
python vectorize_knowledge_base.py
streamlit run main.py
```

### Streamlit Cloud
```bash
# 1. Push to GitHub
git push origin main

# 2. Visit share.streamlit.io
# 3. Connect repository and configure Secrets
# 4. Click Deploy
```

---

## 📚 Next Steps

After successful deployment:

1. **Experience Core Features**: Chat with species
2. **Test RAG Quality**: Run `test_rag_quality.py`
3. **Test Smart Search**: Run `test_smart_search.py`
4. **Read Full Documentation**: [docs/COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md)
5. **Customize Configuration**: Adjust parameters in `.env` 

---

**Happy Deployment!** 🎉

[⬆ Back to Top](#-chatspecies---quick_deploy)

