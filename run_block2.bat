@echo off
REM ================================================================
REM  BLOCK 2: diagnose the visual failure WITHOUT any training
REM  (runs on the block-1 checkpoints, inference only).
REM
REM  A) Open KGT5 protocol (sample eval): none / text / visual, --eval_only.
REM     Removes the "type-constrained rank hides the image" confound:
REM     candidates are NOT type-restricted; gold is ranked among all |E|.
REM     Comparable to KGT5 / KGT5-context tables.
REM  B) image_influence on visual: causal dNLL(real vs blank/noise/shuffle)
REM     + attention mass on visual tokens. Answers: does the image reach
REM     the language model at all on the TRAINED checkpoint.
REM
REM  No training here (inference only) -- each run is a separate process,
REM  VRAM is freed between runs.
REM
REM  RUN from the repo root (myenv auto-activates from ..\myenv):
REM      run_block2.bat
REM  Override interpreter:  set PYTHON=C:\...\python.exe  (then)  run_block2.bat
REM
REM  NOTE: this file is intentionally ASCII-only. Do NOT add "chcp 65001"
REM  or Cyrillic text -- the cmd batch parser desyncs on multibyte chars
REM  under UTF-8 and shreds set/redirect lines.
REM ================================================================
setlocal
cd /d "%~dp0"

REM -- Activate myenv from the parent folder --
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
set LOGDIR=graphvis_ckpts\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM -- parameters (tune as needed) --
REM   EVAL_N  = test queries to score (paired across all three conditions)
REM   NSAMP   = samples per query in KGT5 (500 = as in the paper; 200 = fast pilot)
REM   GENB    = generation batch (12 GB ~25-50); INF_N = examples for probes B/C
set EVAL_N=600
set NSAMP=200
set GENB=32
set INF_N=300

echo ================================================================
echo  BLOCK 2 START: %DATE% %TIME%
echo  python  = %PYTHON%
echo  dataset = %DS%
echo  A) open KGT5 eval none/text/visual (eval_only, NO training)
echo  B) image_influence on visual (causal image contribution)
echo  eval_n=%EVAL_N%  num_samples=%NSAMP%  gen_batch=%GENB%  inf_n=%INF_N%
echo ================================================================

REM -- baseline VRAM (free threshold = baseline + 800 MiB) --
set BASE=0
for /f %%u in ('%PYTHON% scripts\gpu_mem.py --print') do set BASE=%%u
if not defined BASE set BASE=0
set /a THRESH=%BASE%+800
echo Baseline VRAM: %BASE% MiB  (between runs wait until used is below %THRESH% MiB)
echo.

echo [%TIME%] (A 3/3) open eval: context_mode=visual  (log: %LOGDIR%\sample_visual.log)
%PYTHON% scripts\viability_gate.py --eval_only --context_mode visual --eval_mode sample --dataset_root %DS% --train_n 8 --eval_n %EVAL_N% --num_samples %NSAMP% --gen_batch %GENB% --output_dir graphvis_ckpts\ctx_visual --out graphvis_ckpts\report_sample_visual.json > "%LOGDIR%\sample_visual.log" 2>&1
set RC3=%ERRORLEVEL%
echo [%TIME%] visual done (code=%RC3%)
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 180

echo.
echo [%TIME%] paired summary table (open protocol, dMRR-CI + McNemar)
%PYTHON% scripts\compare_conditions.py --kind sample --conditions none,text,visual --baseline none > "%LOGDIR%\compare_sample.log" 2>&1
type "%LOGDIR%\compare_sample.log"

echo.
echo [%TIME%] (B) image_influence on visual  (log: %LOGDIR%\image_influence_visual.log)
%PYTHON% scripts\image_influence.py graphvis_ckpts\ctx_visual\adapter --dataset_root %DS% --n %INF_N% --out graphvis_ckpts\ctx_visual\image_influence.json > "%LOGDIR%\image_influence_visual.log" 2>&1
set RC4=%ERRORLEVEL%
echo [%TIME%] image_influence done (code=%RC4%)

echo ================================================================
echo  BLOCK 2 SUMMARY: %DATE% %TIME%    (code=0 means success)
echo    A none    code=%RC1%   graphvis_ckpts\report_sample_none.json
echo    A text    code=%RC2%   graphvis_ckpts\report_sample_text.json
echo    A visual  code=%RC3%   graphvis_ckpts\report_sample_visual.json
echo    compare (open): graphvis_ckpts\compare_conditions_sample.json
echo    B visual  code=%RC4%   graphvis_ckpts\ctx_visual\image_influence.json
echo    logs: %LOGDIR%\sample_*.log, compare_sample.log, image_influence_visual.log
echo ================================================================
endlocal
pause
