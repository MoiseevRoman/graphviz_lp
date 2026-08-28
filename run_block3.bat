@echo off
REM ================================================================
REM  BLOCK 3: does STAGE-1 graph-comprehension pretraining make the
REM  image start to matter? (tests the "maybe it lacks stage1" idea)
REM
REM  Train stage1 (READ the graph: node/edge counts, degree, node &
REM  triple listings) -> stage2 (link prediction), context=visual,
REM  train_vision, on the SAME model (curriculum). Then RE-RUN the
REM  SAME block-1/2 probes on the new checkpoint so numbers are
REM  directly comparable to the no-stage1 visual run:
REM    1) rank ablation real vs blank  (image contribution, low-var)
REM    2) compare none / text / visual(no-s1) / visual_s1
REM       (paired dMRR 95%% CI + McNemar vs none)
REM    3) image_influence: causal dNLL(real vs blank/noise/shuffle)
REM       + attention mass on visual tokens
REM
REM  Read-out: (a) does visual_s1 finally beat none in compare, and
REM  (b) does the image now causally matter (rank real-blank dMRR in
REM  the train log; image_influence dNLL/attn up vs block-2)?
REM
REM  ASCII-only on purpose: the cmd batch parser desyncs on multibyte
REM  chars under UTF-8. Do NOT add "chcp 65001" or Cyrillic text.
REM
REM  RUN from the repo root (myenv auto-activates from ..\myenv):
REM      run_block3.bat
REM  Override interpreter:  set PYTHON=C:\...\python.exe  (then)  run_block3.bat
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
set OUT=graphvis_ckpts\ctx_visual_s1
set LOGDIR=graphvis_ckpts\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM -- parameters (tune as needed) --
REM   TRAIN_N = stage2 link-prediction training graphs (match block-1 visual = 6000)
REM   EVAL_N  = rank test queries (paired real/blank)
REM   S1_EP   = stage1 epochs
REM   S1_N    = stage1 examples cap. The stage1 set is now LEAN (~7.5k: 2500 random
REM             graphs x 3 reading tasks node_count/node_listing/triple_listing).
REM             0 (or any value >= ~7500) = use ALL of it (~1h stage1 on 12 GB).
REM             Rebuild with build_stage1_from_stage2.py --graphs/--tasks to change
REM             coverage. Lower S1_N only to shrink further.
REM   INF_N   = examples for image_influence probes B/C
set TRAIN_N=6000
set EVAL_N=600
set S1_EP=1
set S1_N=16000
set INF_N=300

echo ================================================================
echo  BLOCK 3 START: %DATE% %TIME%
echo  python  = %PYTHON%
echo  dataset = %DS%
echo  out     = %OUT%   (stage1 -^> stage2 visual, train_vision)
echo  train_n=%TRAIN_N%  eval_n=%EVAL_N%  stage1_epochs=%S1_EP%  stage1_n=%S1_N%  inf_n=%INF_N%
echo  NOTE: stage1 set is LEAN (~7.5k). S1_N=0 or ^>=7500 uses all (~1h stage1).
echo ================================================================

REM -- baseline VRAM (free threshold = baseline + 800 MiB) --
set BASE=0
for /f %%u in ('%PYTHON% scripts\gpu_mem.py --print') do set BASE=%%u
if not defined BASE set BASE=0
set /a THRESH=%BASE%+800
echo Baseline VRAM: %BASE% MiB  (between runs wait until used is below %THRESH% MiB)
echo.

echo [%TIME%] (1/3) TRAIN stage1-^>stage2 visual + rank ablation real,blank
echo         log: %LOGDIR%\block3_train_eval.log
%PYTHON% scripts\viability_gate.py --with_stage1 --stage1_epochs %S1_EP% --stage1_n %S1_N% --context_mode visual --train_vision --eval_mode rank --image_modes real,blank --dataset_root %DS% --train_n %TRAIN_N% --eval_n %EVAL_N% --output_dir %OUT% --out %OUT%\report_rank_visual_s1.json > "%LOGDIR%\block3_train_eval.log" 2>&1
set RC1=%ERRORLEVEL%
echo [%TIME%] train+rank done (code=%RC1%)
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300

echo.
echo [%TIME%] (2/3) compare none/text/visual(no-s1)/visual_s1  (rank, dMRR-CI + McNemar)
%PYTHON% scripts\compare_conditions.py --kind rank --conditions none,text,visual,visual_s1 --baseline none > "%LOGDIR%\block3_compare.log" 2>&1
type "%LOGDIR%\block3_compare.log"

echo.
echo [%TIME%] (3/3) image_influence on stage1 checkpoint  (log: %LOGDIR%\block3_image_influence.log)
%PYTHON% scripts\image_influence.py %OUT%\adapter --dataset_root %DS% --n %INF_N% --out %OUT%\image_influence.json > "%LOGDIR%\block3_image_influence.log" 2>&1
set RC2=%ERRORLEVEL%
echo [%TIME%] image_influence done (code=%RC2%)

echo ================================================================
echo  BLOCK 3 SUMMARY: %DATE% %TIME%    (code=0 means success)
echo    1 train+rank      code=%RC1%   %OUT%\report_rank_visual_s1.json
echo                                   %OUT%\test_details_rank.json       (real)
echo                                   %OUT%\test_details_rank_blank.json (blank)
echo    2 compare (rank): graphvis_ckpts\compare_conditions.json
echo    3 image_influence code=%RC2%   %OUT%\image_influence.json
echo    logs: %LOGDIR%\block3_*.log
echo ================================================================
endlocal
pause
