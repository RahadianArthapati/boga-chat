@echo off
echo Setting up Boga Chat environments...

:: Setup backend
echo Setting up backend environment...
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
echo Backend setup complete!
call venv\Scripts\deactivate.bat
cd ..

:: Setup frontend
echo Setting up frontend environment...
cd frontend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
echo Frontend setup complete!
call venv\Scripts\deactivate.bat
cd ..

echo Setup complete!
echo.
echo To activate the backend environment:
echo   cd backend ^&^& venv\Scripts\activate
echo.
echo To activate the frontend environment:
echo   cd frontend ^&^& venv\Scripts\activate
echo.
echo To start the backend server:
echo   uvicorn app.main:app --reload
echo.
echo To start the frontend app:
echo   streamlit run app.py 