@echo off
REM ================================================================
REM  READABILITY CHECK BEFORE THE BIG-K LP RUN (run_bigk.bat).
REM  Can the BASE model (no training) read K=32 / K=64 star renders?
REM  If it cannot, LP training on them is pointless.
REM
REM  Renders only the first 2000 test queries (the probe takes 100 of
REM  the seed-42 common-test queries), 1-hop star, hybrid ranking,
REM  --max_nodes K+2. Grid:
REM    K=16 @ 560 px  reference (existing graphvis_scaling\k16)
REM    K=32 @ 560 px (400 tok)   K=32 @ 728 px (676 tok)
REM    K=64 @ 560 px (400 tok)   K=64 @ 728 px (676 tok)
REM  Probe: scripts\read_choice_probe.py (Yes/No, AUC real vs shuffle):
REM    node      is the entity box on the image
REM    edge_rel  is the edge label bound to the right box
REM  Reference K=16 base: node 0.998, edge_rel 0.890.
REM
REM  Needs Neo4j (render step only). ~1-1.5 h total.
REM  Output: graphvis_readcheck\s<px>\k<K>\  (test renders)
REM          graphvis_readcheck\read_choice.json, logs\
REM  Finished variants are skipped on re-run. Nothing existing changes.
REM
REM  ASCII-only on purpose. RUN from repo root: run_bigk_readcheck.bat
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
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ================================================================
echo  BIG-K READABILITY CHECK START: %DATE% %TIME%
echo ================================================================

set BASE=0
for /f %%u in ('%PYTHON% scripts\gpu_mem.py --print') do set BASE=%%u
if not defined BASE set BASE=0
set /a THRESH=%BASE%+800

REM ---------- reference: existing K=16 renders ----------
echo [%TIME%] PROBE K=16 @ 560  ^(log: %LOGDIR%\probe_k16_s560.log^)
%PYTHON% scripts\read_choice_probe.py --k 16 --data_base graphvis_scaling --label base_s560 --out %PROBE% > "%LOGDIR%\probe_k16_s560.log" 2>&1
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300

call :CHECK 32 560
call :CHECK 64 728
call :CHECK 64 560
call :CHECK 32 728

echo.
echo ================ SUMMARY ================
%PYTHON% -c "import sys,json;sys.path.insert(0,'scripts');import read_choice_probe as r;r.print_table(json.load(open(r'%PROBE%',encoding='utf-8')))"
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
set DS=%RB%\k%K%\datasets
if exist "%DS%\stage2_test\dataset_info.json" goto C_HAVEDS
%PYTHON% scripts\check_neo4j.py
if errorlevel 1 (
  echo [X] Neo4j is not reachable - start it and re-run. K=%K% @ %SIZE% skipped.
  goto :eof
)
echo [%TIME%] RENDER K=%K% @ %SIZE% px, test 2000  ^(log: %LOGDIR%\build_k%K%_s%SIZE%.log^)
%PYTHON% scripts\rerender_dataset.py --method hybrid --k %K% --target_size %SIZE% --max_nodes %MAXN% --splits stage2_test --limit 2000 --workers 4 --out_root %RB%\k%K% > "%LOGDIR%\build_k%K%_s%SIZE%.log" 2>&1
if not exist "%DS%\stage2_test\dataset_info.json" (
  echo [X] render failed - see %LOGDIR%\build_k%K%_s%SIZE%.log
  goto :eof
)
:C_HAVEDS
echo [%TIME%] PROBE K=%K% @ %SIZE%  ^(log: %LOGDIR%\probe_k%K%_s%SIZE%.log^)
%PYTHON% scripts\read_choice_probe.py --k %K% --data_base %RB% --max_nodes %MAXN% --label base_s%SIZE% --out %PROBE% > "%LOGDIR%\probe_k%K%_s%SIZE%.log" 2>&1
echo [%TIME%] probe done ^(code=!ERRORLEVEL!^)
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300
goto :eof
