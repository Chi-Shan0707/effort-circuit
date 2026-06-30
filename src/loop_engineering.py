from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from datetime import datetime, timezone


def run_loop(args: argparse.Namespace) -> int:
    command = shlex.split(args.command)
    iteration = 0
    while args.iterations == 0 or iteration < args.iterations:
        iteration += 1
        stamp = datetime.now(timezone.utc).isoformat()
        print(f"[{stamp}] loop iteration {iteration}: {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=args.cwd)
        if result.returncode != 0 and args.stop_on_failure:
            return result.returncode
        if args.iterations != 0 and iteration >= args.iterations:
            break
        time.sleep(args.interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an engineering command continuously.")
    parser.add_argument("--command", default="python -m pytest -q")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--iterations", type=int, default=0, help="0 means forever.")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_loop(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
