@echo off
REM ================================================================
REM  DOES STAGE-1 READING TRAINING SHIFT THE LEGIBILITY THRESHOLD?
REM  Follow-up to run_bigk_readcheck.bat (base model). Same Yes/No
REM  probe, same renders, but with the EXISTING stage-1 adapter
REM  (trained on K=16 stars only, no LP):
REM    graphvis_ckpts_s1\k16\ctx_visual_s1\adapter_stage1
REM  Base model needs ~20-25 visual tokens per node. If stage-1 reads
REM  K=64 @ 560 px (~9 tokens/node) at node AUC ~0.95+, the image
REM  becomes cheaper than text (400 vs ~683 tokens) and it is worth
REM  training stage-1 on dense stars. If not - close the budget idea.
REM
REM  Base reference (node / edge_rel AUC):
REM    K32 @ 560  0.903 / 0.711     K32 @ 728  0.983 / 0.804
REM    K64 @ 560  0.849 / 0.720     K64 @ 728  0.909 / 0.774
REM    K16 @ 560  0.998 / 0.890  (stage1 at K16: 1.000 / 0.923)
REM
REM  No training, no Neo4j (renders already in graphvis_readcheck).
REM  ~20-25 min per variant, ~1.5 h total. Results are added to
REM  graphvis_readcheck\read_choice.json under stage1_s<px>/K<K>.
REM  Finished variants are skipped on re-run.
REM
REM  ASCII-only on purpose. RUN from repo root.
REM  Override interpreter: set PYTHON=C:\...\python.exe
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

set RC=graphvis_readcheck
set PROBE=%RC%\read_choice.json
set LOGDIR=%RC%\logs
set A=graphvis_ckpts_s1\k16\ctx_visual_s1\adapter_stage1
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if not exist "%A%\adapter_config.json" (
  echo [X] no stage-1 adapter %A%
  goto END
)

echo ================================================================
echo  STAGE-1 READABILITY CHECK START: %DATE% %TIME%
echo ================================================================

set BASE=0
for /f %%u in ('%PYTHON% scripts\gpu_mem.py --print') do set BASE=%%u
if not defined BASE set BASE=0
set /a THRESH=%BASE%+800

call :CHECK 64 560
call :CHECK 32 560
call :CHECK 64 728
call :CHECK 32 728

echo.
echo ================ SUMMARY (base_* and stage1_*) ================
%PYTHON% -c "import sys,json;sys.path.insert(0,'scripts');import read_choice_probe as r;r.print_table(json.load(open(r'%PROBE%',encoding='utf-8')))"
:END
echo ================================================================
echo  DONE: %DATE% %TIME%   result: %PROBE%
echo ================================================================
endlocal
pause
goto :EOF

REM ================================================================
REM  Subroutines
REM ================================================================
:CHECK
REM  %1 = K   %2 = render size px
set K=%~1
set SIZE=%~2
set /a MAXN=%K%+2
set RB=%RC%\s%SIZE%
if not exist "%RB%\k%K%\datasets\stage2_test\dataset_info.json" (
  echo [!] no renders %RB%\k%K% - run run_bigk_readcheck.bat first. Skipped.
  goto :eof
)
echo [%TIME%] STAGE1 PROBE K=%K% @ %SIZE%  ^(log: %LOGDIR%\probe_s1_k%K%_s%SIZE%.log^)
%PYTHON% scripts\read_choice_probe.py --k %K% --data_base %RB% --max_nodes %MAXN% --adapter "%A%" --label stage1_s%SIZE% --out %PROBE% > "%LOGDIR%\probe_s1_k%K%_s%SIZE%.log" 2>&1
echo [%TIME%] probe done ^(code=!ERRORLEVEL!^)
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300
goto :eof
