@echo off
REM ================================================================
REM  Блок 1: три пилот-дообучения (none / text / visual) ПОДРЯД.
REM  Параллельно нельзя — 3 QLoRA не влезут в 12 ГБ; каждый прогон
REM  обучает + оценивает и освобождает VRAM перед следующим
REM  (каждый шаг — отдельный процесс python).
REM
REM  ЗАПУСК: просто запустите из корня репо (myenv активируется сам из ..\myenv):
REM      run_block1.bat
REM
REM  Переопределить интерпретатор (если myenv в другом месте):
REM      set PYTHON=C:\путь\к\myenv\Scripts\python.exe  (затем)  run_block1.bat
REM ================================================================
setlocal
cd /d "%~dp0"

REM -- Активация myenv из родительской папки (..\myenv) --
if defined PYTHON goto PYSET
if not exist "..\myenv\Scripts\activate.bat" goto NOENV
call "..\myenv\Scripts\activate.bat"
echo Окружение активировано: ..\myenv
goto PYSET
:NOENV
echo [!] ..\myenv\Scripts\activate.bat не найден - использую python из PATH
:PYSET
if "%PYTHON%"=="" set PYTHON=python
set DS=graphvis_fb15k_data_hybrid8_672\datasets
set LOGDIR=graphvis_ckpts\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ================================================================
echo  БЛОК 1 СТАРТ: %DATE% %TIME%
echo  python  = %PYTHON%
echo  dataset = %DS%
echo  логи    = %LOGDIR%\ctx_*.log
echo ================================================================

REM -- базовый уровень VRAM (порог освобождения = базовый + 800 MiB) --
set BASE=0
for /f %%u in ('%PYTHON% scripts\gpu_mem.py --print') do set BASE=%%u
if not defined BASE set BASE=0
set /a THRESH=%BASE%+800
echo Базовый уровень VRAM: %BASE% MiB  (после каждого прогона ждём освобождения ниже %THRESH% MiB)
echo.

echo [%TIME%] (1/3) context_mode=none   (лог: %LOGDIR%\ctx_none.log)
%PYTHON% scripts\viability_gate.py --context_mode none --dataset_root %DS% --train_n 6000 --eval_n 1500 --epochs 2 --output_dir graphvis_ckpts\ctx_none --out graphvis_ckpts\report_ctx_none.json > "%LOGDIR%\ctx_none.log" 2>&1
set RC_NONE=%ERRORLEVEL%
echo [%TIME%] none завершён (код=%RC_NONE%)
echo [%TIME%] проверка освобождения VRAM ...
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 180

echo [%TIME%] (2/3) context_mode=text   (лог: %LOGDIR%\ctx_text.log)
%PYTHON% scripts\viability_gate.py --context_mode text --dataset_root %DS% --train_n 6000 --eval_n 1500 --epochs 2 --output_dir graphvis_ckpts\ctx_text --out graphvis_ckpts\report_ctx_text.json > "%LOGDIR%\ctx_text.log" 2>&1
set RC_TEXT=%ERRORLEVEL%
echo [%TIME%] text завершён (код=%RC_TEXT%)
echo [%TIME%] проверка освобождения VRAM ...
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 180

echo [%TIME%] (3/3) context_mode=visual --train_vision   (лог: %LOGDIR%\ctx_visual.log)
%PYTHON% scripts\viability_gate.py --context_mode visual --train_vision --dataset_root %DS% --train_n 6000 --eval_n 1500 --epochs 2 --output_dir graphvis_ckpts\ctx_visual --out graphvis_ckpts\report_ctx_visual.json > "%LOGDIR%\ctx_visual.log" 2>&1
set RC_VIS=%ERRORLEVEL%
echo [%TIME%] visual завершён (код=%RC_VIS%)
echo [%TIME%] проверка освобождения VRAM ...
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 180

echo ================================================================
echo  ИТОГ: %DATE% %TIME%    (код=0 — успех)
echo    none    код=%RC_NONE%    graphvis_ckpts\report_ctx_none.json
echo    text    код=%RC_TEXT%    graphvis_ckpts\report_ctx_text.json
echo    visual  код=%RC_VIS%     graphvis_ckpts\report_ctx_visual.json
echo    логи всех прогонов: %LOGDIR%\
echo ================================================================
endlocal
pause
