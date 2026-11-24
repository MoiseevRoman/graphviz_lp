@echo off
REM ================================================================
REM  RE-RUN diverged K=16 text_matrix cell.
REM  Original run diverged (train loss_last = 2.13e6) -> its 0.394 is a
REM  broken-checkpoint number, not a real "matrix fails at K16" result.
REM
REM  Two attempts in ONE run so we end with a stable, usable number:
REM    A) tag=s1     LR 2e-4, seed 1   (PREFERRED: LR stays matched to the
REM                                     other conditions; new seed just changes
REM                                     batch order -> tests if divergence was a
REM                                     stochastic bad-batch fluke)
REM    B) tag=lr1e4  LR 1e-4, seed 42  (FALLBACK if A still diverges; needs a
REM                                     footnote since LR differs)
REM
REM  Trains on graphvis_scaling\k16\datasets (already has neighbors_matrix ->
REM  NO rerender, NO Neo4j). Each run also evals on the COMMON test
REM  (eval_n=1000, rank) so report.json.eval.MRR is directly comparable to the
REM  paper table. Original ctx_text_matrix is left untouched; results go to
REM  ctx_text_matrix_s1 / ctx_text_matrix_lr1e4.
REM
REM  Read-out: loss_last should be ~1.5 (NOT 1e5+). Keep the stable attempt
REM  with the higher MRR; if A is stable, prefer it (matched hyperparams).
REM  Cost: ~2 short 4k trainings + evals (~1h), ~1.3 GB adapters.
REM
REM  ASCII-only on purpose (cmd parser desyncs on multibyte UTF-8).
REM  RUN from repo root:  run_fix_text_matrix_k16.bat
REM  Override interpreter:  set PYTHON=C:\...\python.exe
REM ================================================================
setlocal enabledelayedexpansion
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

set MODEL=Qwen/Qwen2.5-VL-3B-Instruct
set DS=graphvis_scaling\k16\datasets
set CKROOT=graphvis_ckpts_scaling\k16
set LOGDIR=%CKROOT%\logs_fix_text_matrix
set TRAIN_N=4000
set EVAL_N=1000

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ================================================================
echo  FIX K16 text_matrix START: %DATE% %TIME%
echo  python = %PYTHON%
echo  data   = %DS%
echo  out    = %CKROOT%\ctx_text_matrix_{s1,lr1e4}
echo ================================================================

if not exist "%DS%\stage2_train" (
  echo [X] dataset missing: %DS%\stage2_train  - nothing to train on. Aborting.
  goto END
)

REM -- baseline VRAM (free threshold = baseline + 800 MiB) --
set BASE=0
for /f %%u in ('%PYTHON% scripts\gpu_mem.py --print') do set BASE=%%u
if not defined BASE set BASE=0
set /a THRESH=%BASE%+800
echo Baseline VRAM: %BASE% MiB  (between runs wait until below %THRESH% MiB)
echo.

call :RUN s1    2e-4 1
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300
call :RUN lr1e4 1e-4 42
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300

echo.
echo ================================================================
echo  RESULTS (loss_last must be ~1.5, NOT 1e5+; old broken MRR was 0.394)
echo ================================================================
%PYTHON% scripts\show_report_mrr.py %CKROOT%\ctx_text_matrix_s1\report.json %CKROOT%\ctx_text_matrix_lr1e4\report.json

:END
echo ================================================================
echo  DONE: %DATE% %TIME%
echo    reports: %CKROOT%\ctx_text_matrix_s1\report.json
echo             %CKROOT%\ctx_text_matrix_lr1e4\report.json
echo    logs:    %LOGDIR%\
echo  Next: keep the stable attempt (loss ~1.5). If s1 is stable, prefer it.
echo ================================================================
endlocal
pause
goto :EOF

REM ================================================================
:RUN
REM  %1 = tag   %2 = learning_rate   %3 = seed
set TAG=%~1
set LR=%~2
set SEED=%~3
echo [%TIME%] TRAIN text_matrix K16  tag=%TAG% lr=%LR% seed=%SEED%  (log: %LOGDIR%\train_%TAG%.log)
%PYTHON% scripts\viability_gate.py --model %MODEL% --context_mode text_matrix --train_vision --dataset_root %DS% --data_dir FB15k-237 --train_n %TRAIN_N% --eval_n %EVAL_N% --epochs 1 --eval_mode rank --cand_cap 50 --learning_rate %LR% --seed %SEED% --output_dir %CKROOT%\ctx_text_matrix_%TAG% --out %CKROOT%\ctx_text_matrix_%TAG%\report.json > "%LOGDIR%\train_%TAG%.log" 2>&1
echo [%TIME%] %TAG% done (code=%ERRORLEVEL%)
goto :eof
