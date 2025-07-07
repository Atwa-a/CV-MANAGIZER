@echo off
echo === Deleting old virtual environment...
rmdir /s /q venv

echo === Creating new virtual environment...
python -m venv venv

echo === Activating virtual environment...
call venv\Scripts\activate.bat

echo === Upgrading pip...
python -m pip install --upgrade pip

echo === Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo === Done. You can now run the app with:
echo     venv\Scripts\activate
echo     python app.py
pause
