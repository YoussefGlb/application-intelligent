@echo off
cd /d "%~dp0"
tailwindcss.exe -i static\css\input.css -o static\dist\css\style.css --watch
pause
