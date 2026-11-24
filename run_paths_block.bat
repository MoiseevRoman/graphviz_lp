@echo off
REM ================================================================
REM  PATH-AWARE SUBGRAPHS: does the image help when it carries
REM  candidate-aware 2-hop structure (head -> x -> candidate)?
REM
REM  Same protocol as the paper: Qwen2.5-VL-3B, QLoRA, iso-example
REM  (4000 train examples, 1 epoch per condition), rank eval on the
REM  SAME 1000 seed-sampled test queries (971 evaluated), cand_cap=50.
REM  max_length=2048 for ALL conditions here (path prompts are longer;
REM  1024 would cut answers of long examples, cf. text_matrix@K16).
REM
REM  Conditions (scripts\viability_gate.py --context_mode):
REM    none          query only (baseline: what the model knows anyway)
REM    text_cn       counts of 2-hop paths head->candidate (~100 tok)
REM    text_paths    edges of the path subgraph as text (matched info)
REM    visual_paths  rendered path subgraph (560x560, 400 vis. tokens)
REM    both_paths    image + edges as text
REM  visual_paths / both_paths are also evaluated with a BLANK image
REM  (image_modes real,blank): direct test whether the image is used.
REM
REM  Read-out:
REM    text_paths > none           -> path structure helps at all
REM    visual_paths vs text_paths  -> modality, same information
REM    both_paths(real) vs (blank) -> does the image add over text
REM    text_cn vs text_paths       -> are precomputed counts enough
REM
REM  Dataset: graphvis_paths\datasets (built without Neo4j by
REM  scripts\build_path_dataset.py; rebuilt here only if missing).
REM  Writes only into graphvis_ckpts_paths\ ; nothing existing touched.
REM  Disk: ~1.5 GB adapters. Finished conditions are skipped on re-run.
REM
REM  ASCII-only on purpose (cmd parser desyncs on multibyte UTF-8).
REM  RUN from repo root (myenv auto-activates from ..\myenv):
REM      run_paths_block.bat
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
set DATA_ROOT=graphvis_paths
set DS=%DATA_ROOT%\datasets
set CK=graphvis_ckpts_paths
set LOGDIR=%CK%\logs
set TRAIN_N=4000
set EVAL_N=1000
set MAXLEN=2048

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ================================================================
echo  PATH BLOCK START: %DATE% %TIME%
echo  python = %PYTHON%
echo  data   = %DS%
echo  ckpts  = %CK%
echo  train_n=%TRAIN_N%  eval_n=%EVAL_N%  max_length=%MAXLEN%
echo ================================================================

REM ---------- 1) dataset (normally already built) ----------
if exist "%DS%\stage2_test" (
  echo [%TIME%] test split exists: %DS%\stage2_test
) else (
  echo [%TIME%] BUILD test split ^(2000 rows^) -^> %DS%\stage2_test
  %PYTHON% scripts\build_path_dataset.py --splits stage2_test --limit 2000 --workers 4 --out_root %DATA_ROOT% > "%LOGDIR%\build_test.log" 2>&1
  if errorlevel 1 ( echo [X] test build FAILED - see %LOGDIR%\build_test.log & goto END )
)
if exist "%DS%\stage2_train" (
  echo [%TIME%] train split exists: %DS%\stage2_train
) else (
  echo [%TIME%] BUILD train split -^> %DS%\stage2_train
  %PYTHON% scripts\build_path_dataset.py --splits stage2_train --workers 4 --out_root %DATA_ROOT% > "%LOGDIR%\build_train.log" 2>&1
  if errorlevel 1 ( echo [X] train build FAILED - see %LOGDIR%\build_train.log & goto END )
)

REM -- baseline VRAM (free threshold = baseline + 800 MiB) --
set BASE=0
for /f %%u in ('%PYTHON% scripts\gpu_mem.py --print') do set BASE=%%u
if not defined BASE set BASE=0
set /a THRESH=%BASE%+800
echo Baseline VRAM: %BASE% MiB  (between runs wait until below %THRESH% MiB)
echo.

REM ---------- 2) train + eval per condition ----------
REM  cheapest / most informative first: if the run is interrupted,
REM  the key pair text_paths vs visual_paths is already done.
call :RUN text_paths   real
call :RUN visual_paths "real,blank"
call :RUN none         real
call :RUN text_cn      real
call :RUN both_paths   "real,blank"

REM ---------- 3) summaries (paired bootstrap + McNemar) ----------
echo [%TIME%] SUMMARY vs none
%PYTHON% scripts\compare_conditions.py --ckpt_root %CK% --conditions none,text_cn,text_paths,visual_paths,both_paths --baseline none --out %CK%\compare_vs_none.json > "%LOGDIR%\compare_vs_none.log" 2>&1
type "%LOGDIR%\compare_vs_none.log"
echo [%TIME%] SUMMARY vs text_paths (same information, different modality)
%PYTHON% scripts\compare_conditions.py --ckpt_root %CK% --conditions text_paths,visual_paths,both_paths,text_cn --baseline text_paths --out %CK%\compare_vs_text_paths.json > "%LOGDIR%\compare_vs_text_paths.log" 2>&1
type "%LOGDIR%\compare_vs_text_paths.log"

:END
echo ================================================================
echo  DONE: %DATE% %TIME%
echo    reports: %CK%\ctx_^<cond^>\report.json
echo    paired:  %CK%\compare_vs_none.json, %CK%\compare_vs_text_paths.json
echo    logs:    %LOGDIR%\
echo ================================================================
endlocal
pause
goto :EOF

REM ================================================================
REM  Subroutines
REM ================================================================
:RUN
REM  %1 = context_mode  %2 = image_modes for eval
set CD=%~1
set IM=%~2
if exist "%CK%\ctx_%CD%\report.json" (
  echo [%TIME%] %CD%: report exists - skip
  goto :eof
)
echo [%TIME%] TRAIN+EVAL context=%CD%  image_modes=%IM%  (log: %LOGDIR%\train_%CD%.log)
%PYTHON% scripts\viability_gate.py --model %MODEL% --context_mode %CD% --train_vision --dataset_root %DS% --data_dir FB15k-237 --train_n %TRAIN_N% --eval_n %EVAL_N% --epochs 1 --eval_mode rank --cand_cap 50 --image_modes %IM% --max_length %MAXLEN% --output_dir %CK%\ctx_%CD% --out %CK%\ctx_%CD%\report.json > "%LOGDIR%\train_%CD%.log" 2>&1
echo [%TIME%] %CD% done (code=%ERRORLEVEL%)
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300
goto :eof
