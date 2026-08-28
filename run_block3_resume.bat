@echo off
REM ================================================================
REM  BLOCK 3 RESUME: continue AFTER stage1+stage2 are already trained.
REM
REM  The trained adapter is at graphvis_ckpts\ctx_visual_s1\adapter
REM  (stage1 -> stage2 visual + train_vision, saved before eval). This
REM  script does ONLY the inference steps that were left, loading that
REM  adapter with --eval_only (NO retraining):
REM    1) rank ablation real vs blank   (--eval_only)
REM    2) compare none / text / visual(no-s1) / visual_s1
REM    3) image_influence (causal dNLL + attention mass)
REM
REM  Safe to run after stopping run_block3.bat mid-eval: eval has no
REM  partial checkpoint, so the rank pass just restarts from scratch,
REM  but the ~3.75h of training is NOT repeated.
REM
REM  ASCII-only on purpose. Do NOT add "chcp 65001" or Cyrillic.
REM  RUN from repo root (myenv auto-activates):  run_block3_resume.bat
REM ================================================================
setlocal
cd /d "%~dp0"

if defined PYTHON goto PYSET
if not exist "..\myenv\Scripts\activate.bat" goto NOENV
call "..\myenv\Scripts\activate.bat"
echo myenv activated: ..\myenv
goto PYSET
:NOENV
echo [!] ..\myenv\Scripts\activate.bat not found - using python from PATH
:PYSET
if "%PYTHON%"=="" set PYTHON=python

set DS=graphvis_fb15k_data_hybrid8_672\datasets
set OUT=graphvis_ckpts\ctx_visual_s1
set LOGDIR=graphvis_ckpts\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM   EVAL_N = rank test queries (paired real/blank). Lower it (e.g. 300)
REM            if you want the resume to finish faster.
REM   INF_N  = examples for image_influence probes B/C
set EVAL_N=600
set INF_N=300

if not exist "%OUT%\adapter\adapter_config.json" goto NOADAPTER

echo ================================================================
echo  BLOCK 3 RESUME START: %DATE% %TIME%
echo  adapter = %OUT%\adapter   (eval-only, no training)
echo  eval_n=%EVAL_N%  inf_n=%INF_N%
echo ================================================================

set BASE=0
for /f %%u in ('%PYTHON% scripts\gpu_mem.py --print') do set BASE=%%u
if not defined BASE set BASE=0
set /a THRESH=%BASE%+800
echo Baseline VRAM: %BASE% MiB  (between runs wait until used below %THRESH% MiB)
echo.

echo [%TIME%] (1/3) rank ablation real,blank on saved adapter (eval_only)
echo         log: %LOGDIR%\block3_resume_eval.log
%PYTHON% scripts\viability_gate.py --eval_only --context_mode visual --eval_mode rank --image_modes real,blank --dataset_root %DS% --eval_n %EVAL_N% --output_dir %OUT% --out %OUT%\report_rank_visual_s1.json > "%LOGDIR%\block3_resume_eval.log" 2>&1
set RC1=%ERRORLEVEL%
echo [%TIME%] rank done (code=%RC1%)
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300

echo.
echo [%TIME%] (2/3) compare none/text/visual(no-s1)/visual_s1  (rank, dMRR-CI + McNemar)
%PYTHON% scripts\compare_conditions.py --kind rank --conditions none,text,visual,visual_s1 --baseline none > "%LOGDIR%\block3_resume_compare.log" 2>&1
type "%LOGDIR%\block3_resume_compare.log"

echo.
echo [%TIME%] (3/3) image_influence on stage1 checkpoint  (log: %LOGDIR%\block3_resume_image_influence.log)
%PYTHON% scripts\image_influence.py %OUT%\adapter --dataset_root %DS% --n %INF_N% --out %OUT%\image_influence.json > "%LOGDIR%\block3_resume_image_influence.log" 2>&1
set RC2=%ERRORLEVEL%
echo [%TIME%] image_influence done (code=%RC2%)

echo ================================================================
echo  BLOCK 3 RESUME SUMMARY: %DATE% %TIME%   (code=0 means success)
echo    1 rank            code=%RC1%   %OUT%\report_rank_visual_s1.json
echo                                   %OUT%\test_details_rank.json / _blank.json
echo    2 compare (rank): graphvis_ckpts\compare_conditions.json
echo    3 image_influence code=%RC2%   %OUT%\image_influence.json
echo    logs: %LOGDIR%\block3_resume_*.log
echo ================================================================
goto END

:NOADAPTER
echo [!] No adapter at %OUT%\adapter - nothing to resume. Run run_block3.bat first.

:END
endlocal
pause
