#!/usr/bin/env python3
"""Проверка/ожидание освобождения видеопамяти между прогонами.

Читает занятость GPU через ``nvidia-smi`` (НЕ инициализирует CUDA, поэтому сам
память не занимает и не мешает измерению). Чистый stdlib, работает на Windows/Linux.

    python scripts/gpu_mem.py --print                       # печатает used MiB (одно число)
    python scripts/gpu_mem.py --status                      # человекочитаемая строка
    python scripts/gpu_mem.py --wait --below 1500 --timeout 120   # ждёт used < 1500 MiB

Коды возврата: 0 — порог достигнут / nvidia-smi недоступен (не блокируем);
1 — таймаут ожидания (память не освободилась) → батник может предупредить.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time


def query(gpu: int = 0):
    """Возвращает {'used','free','total'} в MiB или None, если nvidia-smi нет/ошибка."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, f"--id={gpu}",
             "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, timeout=15,
        ).decode("utf-8", "replace").strip()
    except Exception:
        return None
    line = out.splitlines()[0] if out else ""
    nums = [int(x) for x in re.findall(r"\d+", line)]
    if len(nums) < 3:
        return None
    return {"used": nums[0], "free": nums[1], "total": nums[2]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--print", action="store_true", dest="do_print",
                    help="печатать только used MiB (для захвата базового уровня)")
    ap.add_argument("--status", action="store_true", help="печатать used/free/total")
    ap.add_argument("--wait", action="store_true", help="ждать, пока used < --below")
    ap.add_argument("--below", type=int, default=1500, help="порог used MiB")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--interval", type=int, default=3)
    args = ap.parse_args()

    st = query(args.gpu)
    if st is None:
        print("[gpu_mem] nvidia-smi недоступен — проверку VRAM пропускаю", file=sys.stderr)
        if args.do_print:
            print(0)
        return 0

    if args.do_print:
        print(st["used"])
        return 0

    if not args.wait:
        print(f"[gpu_mem] used={st['used']} MiB  free={st['free']} MiB  total={st['total']} MiB")
        return 0

    start = time.time()
    while True:
        st = query(args.gpu) or st
        elapsed = int(time.time() - start)
        print(f"[gpu_mem] used={st['used']} MiB  free={st['free']} MiB  "
              f"(порог < {args.below}, прошло {elapsed}s)")
        if st["used"] < args.below:
            print(f"[gpu_mem] OK — видеопамять освобождена (used={st['used']} MiB)")
            return 0
        if elapsed >= args.timeout:
            print(f"[gpu_mem] ВНИМАНИЕ: за {args.timeout}s used не опустилась ниже "
                  f"{args.below} MiB (сейчас {st['used']} MiB). Продолжаю.", file=sys.stderr)
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
