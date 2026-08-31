@echo off
chcp 949 > nul
cls
title Biz Forecaster - 소상공인 AI 수요예측 (설치 및 실행)

rem ===== 이 bat 파일이 있는 폴더로 이동 (바탕화면 등 어디서 실행해도 동작) =====
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ========================================================
echo   Biz Forecaster 를 실행할 준비를 하고 있습니다.
echo   처음 실행할 때는 설치에 10~30분 정도 걸립니다.
echo   창을 닫지 말고 잠시만 기다려 주세요!
echo ========================================================
echo.

if not exist "app.py" (
    echo [오류] app.py 를 찾을 수 없습니다.
    echo        현재 위치: %CD%
    echo        이 bat 파일을 app.py 와 같은 폴더에 두고 실행해 주세요.
    goto :fail
)

rem ===== 0. 사용할 Python 찾기 ==============================
rem   prophet / xgboost 는 최신 3.13~3.14 에서 설치가 실패하는 경우가 많아
rem   3.11 -> 3.12 -> 3.10 순서로 먼저 찾는다.
set "PYCMD="

call :try_launcher 3.11
call :try_launcher 3.12
call :try_launcher 3.10

call :try_path "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
call :try_path "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :try_path "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
call :try_path "C:\Python311\python.exe"
call :try_path "C:\Python312\python.exe"
call :try_path "C:\Python310\python.exe"

rem 그래도 없으면 아무 버전이나 사용
call :try_launcher 3
call :try_where

if not defined PYCMD (
    echo [오류] 이 컴퓨터에서 Python 을 찾지 못했습니다.
    echo        https://www.python.org/downloads/release/python-3119/ 에서
    echo        Python 3.11 을 설치할 때 "Add python.exe to PATH" 를 체크해 주세요.
    goto :fail
)

echo [0/3] 사용할 Python:
%PYCMD% -c "import sys; print('      ' + sys.version.split()[0] + '  ' + sys.executable)"
if errorlevel 1 (
    echo [오류] Python 실행에 실패했습니다.
    goto :fail
)
echo.

rem ===== 1. 가상환경 준비 ===================================
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 가상환경을 만드는 중입니다...
    if exist ".venv" rmdir /s /q ".venv"
    %PYCMD% -m venv .venv
    if errorlevel 1 (
        echo [오류] 가상환경 생성에 실패했습니다.
        goto :fail
    )
) else (
    echo [1/3] 가상환경이 이미 있습니다. 건너뜁니다.
)

set "VPY=%CD%\.venv\Scripts\python.exe"

rem ===== 2. 라이브러리 설치 =================================
"%VPY%" -c "import streamlit, pandas, plotly, sklearn, shap" > nul 2>&1
if not errorlevel 1 (
    echo [2/3] 필요한 라이브러리가 이미 설치되어 있습니다. 건너뜁니다.
    goto :run
)

echo [2/3] 필요한 라이브러리를 설치합니다... ^(인터넷 연결 필요^)
echo       설치 진행 상황이 아래에 표시됩니다. 오래 걸려도 정상입니다.
echo.
"%VPY%" -m pip install --upgrade pip setuptools wheel
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [오류] 라이브러리 설치에 실패했습니다.
    echo        위에 표시된 오류 메시지를 확인해 주세요.
    echo        ^(다시 시도하려면 .venv 폴더를 지운 뒤 이 파일을 다시 실행^)
    goto :fail
)

rem ===== 3. 프로그램 실행 ===================================
:run
echo.
echo [3/3] 프로그램을 시작합니다!
echo       잠시 후 인터넷 브라우저가 열립니다.
echo.
echo       - 처음이라면 [회원가입] 탭에서 매장 정보를 입력하세요.
echo       - 이메일 발송 설정이 없으면 인증코드가 화면에 바로 표시됩니다.
echo       - 종료하려면 이 창에서 Ctrl+C 를 누르거나 창을 닫으세요.
echo.
"%VPY%" -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo [오류] 프로그램 실행 중 문제가 발생했습니다.
    goto :fail
)
goto :end

rem ===== 보조 루틴 ==========================================
:try_launcher
if defined PYCMD exit /b
py -%1 -c "import sys" > nul 2>&1
if errorlevel 1 exit /b
set "PYCMD=py -%1"
exit /b

:try_path
if defined PYCMD exit /b
if not exist %1 exit /b
set "PYCMD="%~1""
exit /b

:try_where
if defined PYCMD exit /b
where python > nul 2>&1
if errorlevel 1 exit /b
set "PYCMD=python"
exit /b

:fail
echo.
echo ========================================================
echo   실행에 실패했습니다. 이 메시지를 캡처해서 문의해 주세요.
echo ========================================================
pause
exit /b 1

:end
echo.
echo 프로그램이 종료되었습니다.
pause
