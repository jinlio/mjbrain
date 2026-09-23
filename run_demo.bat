@echo off
rem 一键启动 mjbrain demo：A 推荐服务 / B CDP 捕获(弹浏览器) / C 实时推荐，三个独立窗口。
rem 关闭对应窗口即停止该组件。--dry 只打印解析结果不启动（预检用）。
setlocal

set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

rem python：MJBRRAIN_PY 环境变量可覆盖；否则按常见 conda 安装位置探测，
rem 都没有则退回 PATH 的 python（若服务窗报缺依赖，见 README 快速开始或自设 MJBRRAIN_PY）
set PY=
if defined MJBRRAIN_PY set PY=%MJBRRAIN_PY%
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\mjbrain\python.exe" set PY=%USERPROFILE%\anaconda3\envs\mjbrain\python.exe
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\mjbrain\python.exe" set PY=%USERPROFILE%\miniconda3\envs\mjbrain\python.exe
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\envs\mjbrain\python.exe" set PY=%LOCALAPPDATA%\anaconda3\envs\mjbrain\python.exe
if not defined PY if exist "C:\ProgramData\anaconda3\envs\mjbrain\python.exe" set PY=C:\ProgramData\anaconda3\envs\mjbrain\python.exe
if not defined PY if exist "D:\anaconda\envs\mjbrain\python.exe" set PY=D:\anaconda\envs\mjbrain\python.exe
if not defined PY set PY=python

rem 权重探测：救援包原盘 → 发布包解压位 → gate 兜底（开源版通常只有第二级）
set CKPT=%ROOT%\dist\rescue-20260923\extracted\checkpoints
if not exist "%CKPT%\model.safetensors" set CKPT=%ROOT%\checkpoints\m3-orig-0.7098
if not exist "%CKPT%\model.safetensors" set CKPT=%ROOT%\checkpoints\20260921T184331Z-rlcd-gate
if not exist "%CKPT%\model.safetensors" (
    echo 没找到权重。把 Release zip 解压到 %ROOT%\checkpoints\m3-orig-0.7098\ 后重试。
    echo 按任意键退出...
    pause >nul
    exit /b 1
)

rem 每局一个新帧文件（run1、run2...）：同名 .advise.jsonl 自动记录推荐
set N=1
:find
set FRAMES=%ROOT%\data\raw\ms_frames\run%N%.jsonl
if exist "%FRAMES%" (
    set /a N+=1
    goto :find
)

if /i "%~1"=="--dry" (
    echo PY=%PY%
    echo CKPT=%CKPT%
    echo FRAMES=%FRAMES%
    exit /b 0
)

cd /d "%ROOT%"
start "mjbrain A 推荐服务"        "%PY%" -m advisor.server --ckpt "%CKPT%" --port 8765
start "mjbrain B 捕获 弹浏览器"   "%PY%" scripts\run_capture.py --url https://game.maj-soul.com/1/ --out "%FRAMES%"
start "mjbrain C 实时推荐"        "%PY%" scripts\live_from_capture.py --jsonl "%FRAMES%" --follow

echo 三个窗口已开：A=服务(预热约15秒)  B=捕获(会自动弹浏览器)  C=实时推荐
echo 本局帧记录:     %FRAMES%
echo      推荐记录: %ROOT%\data\raw\ms_frames\run%N%.advise.jsonl
echo 关闭对应窗口即停止该组件。
echo 按任意键退出（三个窗口不受影响，关它们请直接点 X）...
pause >nul
