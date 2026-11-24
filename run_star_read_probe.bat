@echo off
REM ================================================================
REM  READING PROBE ON THE PAPER'S OWN RENDERS (1-hop star, 560 px).
REM  Base Qwen2.5-VL, NO training. Same images as the visual condition
REM  (graphvis_scaling\k<K>\graphs), same seed-42 common-test queries.
REM  Two tasks per image:
REM    names    list every entity name in the boxes (except head)
REM    triples  one line per arrow: "relation | entity"
REM             (binds the edge label to the right box - what link
REM             prediction actually needs)
REM  Ground truth = what is DRAWN (same node cap / label truncation as
REM  the renderer), arrow direction is not scored.
REM  Variants: tower NF4 (as in the paper) and tower bf16.
REM  K=4 and K=16, 100 images each. ~45-60 min total. No Neo4j.
REM
REM  Output: graphvis_ckpts_scaling_common\star_read_probe.json
REM  Finished variants are skipped on re-run.
REM
REM  ASCII-only on purpose. RUN from repo root: run_star_read_probe.bat
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

set OUT=graphvis_ckpts_scaling_common\star_read_probe.json
set LOGDIR=graphvis_ckpts_scaling_common\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo [%TIME%] STAR READ PROBE  (log: %LOGDIR%\star_read_probe.log)
%PYTHON% scripts\star_read_probe.py --ks 4,16 --n 100 --tasks names,triples --quants "nf4,vision_bf16" --out %OUT% > "%LOGDIR%\star_read_probe.log" 2>&1
echo [%TIME%] done (code=%ERRORLEVEL%)
type "%LOGDIR%\star_read_probe.log" | findstr /l /c:"K=" /c:"/K" /c:"saved"

echo ================================================================
echo  result: %OUT%
echo ================================================================
endlocal
pause
