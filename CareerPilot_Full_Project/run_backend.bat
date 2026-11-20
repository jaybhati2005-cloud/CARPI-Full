@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env
echo Add your GEMINI_API_KEY to .env before using Gemini-backed features.
uvicorn backend.main:app --reload
