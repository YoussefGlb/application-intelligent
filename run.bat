@echo off
cd /d "%~dp0"
set PATH=%APPDATA%\Python\Scripts;%PATH%
poetry run flask --app app run --debug
pause
