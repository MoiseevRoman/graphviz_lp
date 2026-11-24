@echo off
REM ================================================================
REM  ZERO-SHOT MATRIX: base Qwen2.5-VL WITHOUT any adapter.
REM  Question: what does the model know from pretraining, and what
REM  does fine-tuning add?  Fills the missing row of the matrix
REM                        no context      with context
REM    no fine-tuning      ctx_zs_none     ctx_zs_<cond>    <- HERE
REM    fine-tuned (QLoRA)  0.386 (paths)   paper results
REM
REM  EVAL ONLY (viability_gate.py --no_adapter), no training. Same
REM  rank protocol as the paper: eval_n=1000 -> the same 971 common
REM  queries, cand_cap=50, same prompts per context_mode, base model
REM  in NF4 exactly as under the adapters.
REM
REM  Order = priority (stop any time, finished ones are kept):
REM    none          query only (K-independent: triples are identical
REM                  for k1/k4/k8/k16 and graphvis_paths)
REM    text  K=16    visual K=16 (real + blank image)
REM  (K=4 and the 2-hop path block were dropped: K=16 is the paper's
REM   key point; zero-shot text at K=4 is already known from
REM   graphvis_ckpts\textlp_upperbound.json)
REM  A condition with report.json is skipped on re-run.
REM
REM  Output: graphvis_ckpts_zeroshot\
REM    ctx_zs_<cond>\                      raw zero-shot runs
REM    k16\                               paired tables:
REM      ctx_zs_*  (zero-shot)  ctx_ft_*  (json copies of fine-tuned)
REM      compare_zs.json         zero-shot conditions vs zs_none
REM      compare_ft_<cond>.json  fine-tuned vs zero-shot, same condition
REM  Old results are only COPIED, never changed.
REM  Time: ~1 h per text eval, ~2 h per image mode; total ~5.5 h.
REM
REM  ASCII-only on purpose. RUN from repo root: run_zeroshot.bat
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

set MODEL=Qwen/Qwen2.5-VL-3B-Instruct
set OUT=graphvis_ckpts_zeroshot
set COMMON=graphvis_ckpts_scaling_common
set FTPATHS=graphvis_ckpts_paths
set LOGDIR=%OUT%\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ================================================================
echo  ZERO-SHOT MATRIX START: %DATE% %TIME%
echo  python = %PYTHON%
echo  out    = %OUT%
echo ================================================================

set BASE=0
for /f %%u in ('%PYTHON% scripts\gpu_mem.py --print') do set BASE=%%u
if not defined BASE set BASE=0
set /a THRESH=%BASE%+800
echo Baseline VRAM: %BASE% MiB  (between runs wait until below %THRESH% MiB)
echo.

call :ZS none          none         graphvis_scaling\k4\datasets  blank        0
call :ZS text_k16      text         graphvis_scaling\k16\datasets blank        0
call :ZS visual_k16    visual       graphvis_scaling\k16\datasets "real,blank" 0

REM ---------- paired tables ----------
call :TABLE k16   %COMMON%\k16 text_k16 visual_k16

echo.
echo ================ SUMMARY ================
for %%t in (k16) do (
  if exist "%LOGDIR%\compare_%%t_zs.log" (
    echo ---------------- %%t: zero-shot, vs zs_none ----------------
    type "%LOGDIR%\compare_%%t_zs.log" | findstr /v /l /c:"[ok]" /c:"[protocol]" /c:"[skip]" /c:"Saved" /c:"json"
    echo ---------------- %%t: fine-tuned vs zero-shot ----------------
    type "%LOGDIR%\compare_%%t_ft.log" | findstr /l /c:"===" /c:"MRR ="
  )
)

echo ================================================================
echo  DONE: %DATE% %TIME%
echo    raw:    %OUT%\ctx_zs_^<cond^>\report.json
echo    tables: %OUT%\k16\compare_zs.json, compare_ft_*.json
echo ================================================================
endlocal
pause
goto :EOF

REM ================================================================
REM  Subroutines
REM ================================================================
:ZS
REM  %1 = run name  %2 = context_mode  %3 = dataset root
REM  %4 = image modes (QUOTED if it has a comma - call splits on commas)
REM  %5 = max_length (0 = default 1024; 2048 as in the path block)
set NAME=%~1
set CM=%~2
set DS=%~3
set IMS=%~4
set ML=%~5
set ZD=%OUT%\ctx_zs_%NAME%
if exist "%ZD%\report.json" (
  echo [%TIME%] zs_%NAME%: report exists - skip
  goto :eof
)
if not exist "%DS%\stage2_test" (
  echo [!] zs_%NAME%: no dataset %DS% - skip
  goto :eof
)
echo [%TIME%] ZERO-SHOT %NAME% context=%CM% images=%IMS%  ^(log: %LOGDIR%\zs_%NAME%.log^)
%PYTHON% scripts\viability_gate.py --no_adapter --model %MODEL% --context_mode %CM% --dataset_root %DS% --data_dir FB15k-237 --eval_n 1000 --eval_mode rank --cand_cap 50 --image_modes "%IMS%" --max_length %ML% --output_dir %ZD% --out %ZD%\report.json > "%LOGDIR%\zs_%NAME%.log" 2>&1
echo [%TIME%] zs_%NAME% done ^(code=!ERRORLEVEL!^)
REM  blank-image details of visual runs -> own condition dir for compare
if "%IMS%"=="real,blank" if exist "%ZD%\test_details_rank_blank.json" (
  if not exist "%ZD%_blank" mkdir "%ZD%_blank"
  copy /y "%ZD%\test_details_rank_blank.json" "%ZD%_blank\" >nul
)
%PYTHON% scripts\gpu_mem.py --wait --below %THRESH% --timeout 300
goto :eof

:TABLE
REM  %1 = table name  %2 = fine-tuned source root  %3 = text run  %4 = visual run
set TN=%~1
set FT=%~2
set TR=%~3
set VR=%~4
set TD=%OUT%\%TN%
if not exist "%OUT%\ctx_zs_none\report.json" (
  echo [!] %TN%: zs_none missing - table skipped
  goto :eof
)
if not exist "%TD%" mkdir "%TD%"
REM  zero-shot runs: ctx_zs_none, ctx_zs_text, ctx_zs_visual, ctx_zs_visual_blank
for %%p in ("none:none" "text:%TR%" "visual:%VR%" "visual_blank:%VR%_blank") do (
  for /f "tokens=1,2 delims=:" %%a in (%%p) do (
    if exist "%OUT%\ctx_zs_%%b" (
      if not exist "%TD%\ctx_zs_%%a" mkdir "%TD%\ctx_zs_%%a"
      copy /y "%OUT%\ctx_zs_%%b\test_details_rank*.json" "%TD%\ctx_zs_%%a\" >nul
    )
  )
)
REM  fine-tuned copies: none from the path block (same protocol, 971 queries)
if not exist "%TD%\ctx_ft_none" mkdir "%TD%\ctx_ft_none"
copy /y "%FTPATHS%\ctx_none\test_details_rank*.json" "%TD%\ctx_ft_none\" >nul
if "%TN%"=="paths" (
  set FTT=text_paths
  set FTV=visual_paths
) else (
  set FTT=text
  set FTV=visual
)
if exist "%FT%\ctx_!FTT!" (
  if not exist "%TD%\ctx_ft_text" mkdir "%TD%\ctx_ft_text"
  copy /y "%FT%\ctx_!FTT!\test_details_rank*.json" "%TD%\ctx_ft_text\" >nul
)
if exist "%FT%\ctx_!FTV!" (
  if not exist "%TD%\ctx_ft_visual" mkdir "%TD%\ctx_ft_visual"
  copy /y "%FT%\ctx_!FTV!\test_details_rank.json" "%TD%\ctx_ft_visual\" >nul 2>&1
  copy /y "%FT%\ctx_!FTV!\test_details_rank_real.json" "%TD%\ctx_ft_visual\" >nul 2>&1
)
echo [%TIME%] TABLE %TN%
%PYTHON% scripts\compare_conditions.py --ckpt_root %TD% --conditions zs_none,zs_text,zs_visual,zs_visual_blank --baseline zs_none --out %TD%\compare_zs.json > "%LOGDIR%\compare_%TN%_zs.log" 2>&1
%PYTHON% scripts\compare_conditions.py --ckpt_root %TD% --conditions zs_none,ft_none --baseline zs_none --out %TD%\compare_ft_none.json > "%LOGDIR%\compare_%TN%_ft_none.log" 2>&1
%PYTHON% scripts\compare_conditions.py --ckpt_root %TD% --conditions zs_text,ft_text --baseline zs_text --out %TD%\compare_ft_text.json > "%LOGDIR%\compare_%TN%_ft_text.log" 2>&1
%PYTHON% scripts\compare_conditions.py --ckpt_root %TD% --conditions zs_visual,ft_visual --baseline zs_visual --out %TD%\compare_ft_visual.json > "%LOGDIR%\compare_%TN%_ft_visual.log" 2>&1
type "%LOGDIR%\compare_%TN%_ft_none.log" "%LOGDIR%\compare_%TN%_ft_text.log" "%LOGDIR%\compare_%TN%_ft_visual.log" > "%LOGDIR%\compare_%TN%_ft.log" 2>nul
goto :eof
