"""Start the local backend+CPU worker and Vite; Ctrl-C stops all children."""
import os
import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hybrid',action='store_true',help='Enable experimental pretrained CPU lens proposals for front analysis')
    parser.add_argument('--sam2',action='store_true',help='Use local SAM 2.1 with automatic prompts for front analysis')
    parser.add_argument('--sam3',action='store_true',help='Use official SAM 3 eyewear text prompts on local CPU')
    args=parser.parse_args()
    if sum((args.hybrid,args.sam2,args.sam3))>1:
        parser.error('Choose one of --sam3, --sam2 or --hybrid')
    if args.sam3:
        os.environ['EYEWEAR_DETECTION_BACKEND']='sam3'
    elif args.sam2:
        os.environ['EYEWEAR_DETECTION_BACKEND']='sam2'
    elif args.hybrid:
        os.environ['EYEWEAR_DETECTION_BACKEND']='hybrid'
    os.chdir(ROOT)
    python=ROOT/'.venv/bin/python'
    if not python.exists() or not (ROOT/'node_modules').exists():
        raise SystemExit('Install dependencies first; see README.md.')
    processes=[]
    names=['API / worker (8018)','frontend (5188)']
    def stop(*_,exit_code=0):
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        sys.exit(exit_code)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        processes.append(subprocess.Popen([str(python),'-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8018']))
        processes.append(subprocess.Popen(['npm','run','dev']))
    except OSError as error:
        print(f'Frame Lab could not start: {error}',file=sys.stderr,flush=True)
        stop(exit_code=1)
    print('Starting Frame Lab: http://127.0.0.1:5188 · API: http://127.0.0.1:8018/docs\nKeep this terminal open. If either service exits, both are stopped; the error is printed above the shutdown message.',flush=True)
    print('Detection backend: '+os.environ.get('EYEWEAR_DETECTION_BACKEND','heuristics'),flush=True)
    while True:
        failed=next(((name,p.returncode) for name,p in zip(names,processes) if p.poll() is not None),None)
        if failed:
            print(f'Frame Lab stopped: {failed[0]} exited with code {failed[1]}.\nCheck that service\'s error above. EPERM means permission denied; EADDRINUSE means a port is occupied. Restart with .venv/bin/python scripts/dev.py after resolving it.',file=sys.stderr,flush=True)
            stop(exit_code=failed[1] if failed[1] and failed[1]>0 else 1)
        time.sleep(.5)

if __name__=='__main__':main()
