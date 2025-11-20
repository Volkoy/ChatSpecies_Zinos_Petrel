@echo off
REM ==========================================
REM   RAG System Quick Setup Script
REM ==========================================

echo ==========================================
echo   RAG System Quick Setup
echo ==========================================
echo.

REM 1. Install dependencies
echo [1/3] Install dependency packages...
pip install -q tqdm
echo ✅ Completed
echo.

REM 2. Run the vectorized script
echo [2/3] Start vectorization (5-10 minutes)...
echo.
python vectorize_knowledge_base.py
echo.

REM 3. Complete
echo [3/3] Setup complete!
echo.
echo ==========================================
echo   🎉 RAG system is ready!
echo ==========================================
echo.
echo Next Step: 
echo   Run 'streamlit run main.py' to get started.
echo.
pause

