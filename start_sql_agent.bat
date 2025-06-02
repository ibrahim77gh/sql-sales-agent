@echo off
cd /d C:\Work\sql-agent\sql-agent-v1

:: Activate virtual environment
call venv\Scripts\activate

:: Start Streamlit app in a new command window
start "Streamlit" cmd /k "streamlit run streamlit_app.py"

:: Wait a few seconds to let Streamlit start
timeout /t 5 /nobreak > nul

:: Start ngrok in a new command window
start "ngrok" cmd /k "ngrok http 8501"
