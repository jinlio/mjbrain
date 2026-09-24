@echo off
rem mjbrain one-key demo launcher (Windows). Thin wrapper: all real logic is
rem in scripts\run_demo.py (cross-platform: advisor + capture + live + HUD,
rem persistent browser profile, --app 1600x900 window, run*.jsonl rotation,
rem ckpt chain probing). Just pass through all args.
rem Interpreter: override with MJABRAIN_PY env (note: correct spelling;
rem the old var name MJBRRAIN_PY (typo) is still honored for compatibility),
rem otherwise probe common conda locations, else PATH python.
setlocal
set "PY="
if defined MJABRAIN_PY set "PY=%MJABRAIN_PY%"
if defined MJBRRAIN_PY if not defined PY set "PY=%MJBRRAIN_PY%"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\mjbrain\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\mjbrain\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\mjbrain\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\mjbrain\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\envs\mjbrain\python.exe" set "PY=%LOCALAPPDATA%\anaconda3\envs\mjbrain\python.exe"
if not defined PY if exist "C:\ProgramData\anaconda3\envs\mjbrain\python.exe" set "PY=C:\ProgramData\anaconda3\envs\mjbrain\python.exe"
if not defined PY if exist "D:\anaconda\envs\mjbrain\python.exe" set "PY=D:\anaconda\envs\mjbrain\python.exe"
if not defined PY set "PY=python"

"%PY%" "%~dp0scripts\run_demo.py" %*
if errorlevel 1 pause
