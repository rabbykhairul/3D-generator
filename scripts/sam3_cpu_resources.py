"""Bound the local SAM 3 worker's scheduling priority and process memory."""
import os
import sys
import threading


def protect_cpu_worker():
    import psutil

    if hasattr(os, 'nice'):
        try:
            os.nice(10)
        except OSError:
            pass
    limit = min(5 * 1024**3, int(psutil.virtual_memory().total * .625))
    process = psutil.Process()
    stop = threading.Event()
    peak = [0]

    def watch():
        while not stop.is_set():
            rss = process.memory_info().rss
            peak[0] = max(peak[0], rss)
            if rss > limit:
                print('SAM 3 stopped: CPU process memory exceeded its '
                      f'{limit / 1024**3:.1f} GiB limit.', file=sys.stderr, flush=True)
                os._exit(2)
            stop.wait(1)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return stop, peak
