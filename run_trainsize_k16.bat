@echo off
REM ================================================================
REM  TRAIN-SIZE ABLATION, K=16
REM  Question: is "text >= visual" just an undertraining artifact?
REM  Method: train ONE 16k run per modality (text, visual) with
REM  intermediate checkpoints, then eval each checkpoint on the SAME
REM  1000 test triples -> MRR(n) curve for n in {4k, 8k, 12k, 16k}.
REM
REM  eff.batch = 1 x 8 = 8 examples/step; 16000 ex x 1 epoch = 2000
REM  steps. save_steps=500 -> checkpoint-500/1000/1500/2000, i.e.
REM  n = 4k/8k/12k/16k. The 16k point is the train run's own final
REM  eval (ctx_<cond>\report.json). Read-out: if visual keeps rising
REM  toward 16k while text has plateaued -> visual undertrained (4k
REM  comparison unfair). If both flatten and text stays above -> the
REM  conclusion is robust.
REM
REM  Writes into FRESH trees; nothing existing is touched:
REM    data:   graphvis_scaling_trainsize\k16\...
REM    ckpts:  graphvis_ckpts_trainsize\k16\...
REM  Disk: ~5-6 GB checkpoints (save_only_model) + ~0.7 GB renders.
REM
REM  ASCII-only on purpose (cmd parser desyncs on multibyte UTF-8).
REM  RUN from repo root (myenv auto-activates from ..\myenv):
REM      run_trainsize_k16.bat
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

REM -- parameters --
set MODEL=Qwen/Qwen2.5-VL-3B-Instruct
set DATA_ROOT=graphvis_scaling_trainsize\k16
set DS=%DATA_ROOT%\datasets
set CK=graphvis_ckpts_trainsize\k16
set LOGDIR=%CK%\logs
set K=16
set TS=560
set TRAIN_N=16000
set TEST_N=2000
set EVAL_N=1000
set SAVE_STEPS=500
set SAVE_LIMIT=20

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ================================================================
echo  TRAIN-SIZE ABLATION K=%K% START: %DATE% %TIME%
echo  python = %PYTHON%
echo  data   = %DS%
echo  ckpts  = %CK%
echo  train_n=%TRAIN_N%  eval_n=%EVAL_N%  save_steps=%SAVE_STEPS%
echo ================================================================

REM ---------- 0) preflight: Neo4j must be up (rerender pulls subgraphs) ----------
REM  Skip the check if both splits already exist (no build needed).
if exist "%DS%\stage2_train" if exist "%DS%\stage2_test" goto BUILT
echo [%TIME%] check Neo4j (bolt) before building...
%PYTHON% scripts\check_neo4j.py
if errorlevel 1 (
  echo [X] Neo4j unreachable - start it ^(with FB15k-237 loaded^) and re-run. Aborting.
  goto END
)
:BUILT

REM ---------- 1) build dataset (16k train + 2k test), fresh tree ----------
if exist "%DS%\stage2_train" (
  echo [%TIME%] train split exists, skip build: %DS%\stage2_train
) else (
  echo [%TIME%] BUILD train split -^> %DS%\stage2_train  ^(limit %TRAIN_N%^)
  %PYTHON% scripts\rerender_dataset.py --method hybrid --k %K% --target_size %TS% --splits stage2_train --limit %TRAIN_N% --workers 4 --out_root %DATA_ROOT% > "%LOGDIR%\build_train.log" 2>&1
  if errorlevel 1 ( echo [X] train build FAILED - see %LOGDIR%\build_train.log & goto END )
)
if exist "%DS%\stage2_test" (
  echo [%TIME%] test split exists, skip build: %DS%\stage2_test
) else (
  echo [%TIME%] BUILD test split -^> %DS%\stage2_test  ^(limit %TEST_N%^)
  %PYTHON% scripts\rerender_dataset.py --method hybrid --k %K% --target_size %TS% --splits stage2_test --limit %TEST_N% --workers 4 --out_root %DATA_ROOT% > "%LOGDIR%\build_test.log" 2>&1
  if errorlevel 1 ( echo [X] test build FAILED - see %LOGDIR%\build_test.log & goto END )
)

REM -- baseline VRAM (free threshold = baseline + 800 MiB) --
set BASE=0
for /f %%u in ('%PYTHON% scripts\gpu_mem.py --print') do set BASE=%%u
if not defined BASE set BASE=0
set /a THRESH=%BASE%+800
echo Baseline VRAM: %BASE% MiB  (between runs wait until below %THRESH% MiB)
echo.

REM ---------- 2) train 16k per condition, save checkpoints ----------
call :TRAIN16 text
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300
call :TRAIN16 visual
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300

REM ---------- 3) eval intermediate checkpoints (4k/8k/12k) ----------
REM   checkpoint-500 -> n=4000, -1000 -> n=8000, -1500 -> n=12000
REM   (n=16000 is the train run's own final eval in ctx_<cond>\report.json)
call :EVALONE text   500  04000
call :EVALONE text   1000 08000
call :EVALONE text   1500 12000
call :EVALONE visual 500  04000
call :EVALONE visual 1000 08000
call :EVALONE visual 1500 12000

REM ---------- 4) summarize MRR(n) ----------
echo [%TIME%] SUMMARY
%PYTHON% scripts\summarize_trainsize.py --root %CK% --final_n %TRAIN_N% --out %CK%\trainsize_summary.csv
if exist "%CK%\trainsize_summary.csv" type "%CK%\trainsize_summary.csv"

:END
echo ================================================================
echo  DONE: %DATE% %TIME%
echo    curve:  %CK%\trainsize_summary.csv
echo    reports: %CK%\ctx_text\report.json (16k), %CK%\ctx_text\n04000\report.json ...
echo    logs:   %LOGDIR%\
echo ================================================================
endlocal
pause
goto :EOF

REM ================================================================
REM  Subroutines
REM ================================================================
:TRAIN16
REM  %1 = context_mode (text|visual)
set CD=%~1
echo [%TIME%] TRAIN 16k context=%CD%  (log: %LOGDIR%\train_%CD%.log)
%PYTHON% scripts\viability_gate.py --model %MODEL% --context_mode %CD% --train_vision --dataset_root %DS% --data_dir FB15k-237 --train_n %TRAIN_N% --eval_n %EVAL_N% --epochs 1 --eval_mode rank --cand_cap 50 --save_steps %SAVE_STEPS% --save_total_limit %SAVE_LIMIT% --save_only_model --output_dir %CK%\ctx_%CD% --out %CK%\ctx_%CD%\report.json > "%LOGDIR%\train_%CD%.log" 2>&1
echo [%TIME%] train %CD% done (code=%ERRORLEVEL%)
goto :eof

:EVALONE
REM  %1 = context_mode  %2 = checkpoint step  %3 = n-label
set CD=%~1
set STEP=%~2
set NN=%~3
if not exist "%CK%\ctx_%CD%\viability\checkpoint-%STEP%" (
  echo [!] missing checkpoint-%STEP% for %CD% - skip n=%NN%
  goto :eof
)
echo [%TIME%] EVAL context=%CD% n=%NN%  (checkpoint-%STEP%)
%PYTHON% scripts\viability_gate.py --eval_only --adapter_dir %CK%\ctx_%CD%\viability\checkpoint-%STEP% --model %MODEL% --context_mode %CD% --dataset_root %DS% --data_dir FB15k-237 --eval_mode rank --eval_n %EVAL_N% --cand_cap 50 --output_dir %CK%\ctx_%CD%\n%NN% --out %CK%\ctx_%CD%\n%NN%\report.json > "%LOGDIR%\eval_%CD%_n%NN%.log" 2>&1
echo [%TIME%] eval %CD% n=%NN% done (code=%ERRORLEVEL%)
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300
goto :eof
