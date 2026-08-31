@echo off
chcp 949 > nul
cls
title Biz Forecaster - 이메일 설정 확인
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
    echo [오류] 가상환경이 없습니다.
    echo        먼저 실행하기.bat 을 한 번 실행해 주세요.
    echo.
    pause
    exit /b 1
)

echo.
echo  1 = 설정 확인만 (메일 발송 안 함)
echo  2 = 테스트 메일 실제로 보내기
echo.
set /p MODE=  번호를 입력하세요 (1 또는 2):

if "%MODE%"=="2" (
    ".venv\Scripts\python.exe" check_email.py --send
) else (
    ".venv\Scripts\python.exe" check_email.py
)
