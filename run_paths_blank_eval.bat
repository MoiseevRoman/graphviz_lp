@echo off
REM ================================================================
REM  BLANK-IMAGE EVAL for the path block (eval only, no training).
REM  run_paths_block.bat lost the "blank" mode: cmd split the
REM  unquoted call argument real,blank on the comma, so visual_paths
REM  (and both_paths) were evaluated with the real image only.
REM  Here the SAME adapters are evaluated with a white image on the
REM  SAME 971 queries -> direct test whether the image is used.
REM  Writes to ctx_<cond>_blank\ ; existing reports are not touched.
REM  Run AFTER run_paths_block.bat has finished both_paths.
REM  ASCII-only on purpose.
REM ================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
if defined PYTHON goto PYSET
if exist "..\myenv\Scripts\activate.bat" call "..\myenv\Scripts\activate.bat"
:PYSET
if "%PYTHON%"=="" set PYTHON=python

set MODEL=Qwen/Qwen2.5-VL-3B-Instruct
set DS=graphvis_paths\datasets
set CK=graphvis_ckpts_paths
set LOGDIR=%CK%\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

call :BLANK visual_paths
call :BLANK both_paths

echo [%TIME%] SUMMARY real vs blank
%PYTHON% scripts\compare_conditions.py --ckpt_root %CK% --conditions visual_paths,visual_paths_blank,both_paths,both_paths_blank,text_paths --baseline visual_paths_blank --out %CK%\compare_blank.json > "%LOGDIR%\compare_blank.log" 2>&1
type "%LOGDIR%\compare_blank.log"
endlocal
pause
goto :EOF

:BLANK
set CD=%~1
if not exist "%CK%\ctx_%CD%\adapter" ( echo [!] no adapter for %CD% - skip & goto :eof )
if exist "%CK%\ctx_%CD%_blank\report.json" ( echo [%TIME%] %CD%_blank exists - skip & goto :eof )
echo [%TIME%] EVAL blank image: %CD%
%PYTHON% scripts\viability_gate.py --eval_only --adapter_dir %CK%\ctx_%CD%\adapter --model %MODEL% --context_mode %CD% --dataset_root %DS% --data_dir FB15k-237 --eval_mode rank --eval_n 1000 --cand_cap 50 --image_modes blank --max_length 2048 --output_dir %CK%\ctx_%CD%_blank --out %CK%\ctx_%CD%_blank\report.json > "%LOGDIR%\eval_%CD%_blank.log" 2>&1
echo [%TIME%] %CD%_blank done (code=%ERRORLEVEL%)
goto :eof
