"""Orchestrator: capture each 100k checkpoint of the running training and run the
full magnetic battery on it, then build the comparison figures.

Why an orchestrator and not a shell watcher: the first attempt used `nohup ... &`
and the process was gone within a minute (checked with ps).  This runs as a
single long-lived process that (a) polls loss.out, (b) copies nep.txt the moment
the generation reaches a boundary, (c) immediately runs model_series.run_one on
it, and (d) writes the figures once everything is done.

usage: python series_orchestrator.py [--stop-after 500000]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                                "validation", "BCC_Fe_full_physics", "scripts"))
sys.path.insert(0, _HERE)
import feutils as U
import model_series as MS

RUN = os.path.join(U.ROOT, "work_20261101_retrain", "C_5desc_500k")
CKPT = os.path.join(RUN, "checkpoints")
BOUNDS = [200000, 300000, 400000, 500000]
LOG = os.path.join(CKPT, "orchestrator.log")
TRIES = {}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def gen_now():
    p = os.path.join(RUN, "loss.out")
    try:
        last = open(p).read().strip().splitlines()[-1]
        return int(last.split()[0])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=20)
    A = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    log("orchestrator start")

    # the 110k checkpoint is already on disk: run it first so there is a result
    # even if the session ends before the later boundaries
    p110 = os.path.join(CKPT, "gen_110k.nep.txt")
    if os.path.exists(p110):
        MS.run_one("gen_110k", p110)

    while True:
        for b in BOUNDS:
            tag = f"gen_{b // 1000}k"
            f = os.path.join(CKPT, tag + ".nep.txt")
            if os.path.exists(f):
                continue
            g = gen_now()
            if g >= b:
                time.sleep(8)          # let nep finish writing nep.txt
                shutil.copyfile(os.path.join(RUN, "nep.txt"), f)
                shutil.copyfile(os.path.join(RUN, "loss.out"),
                                os.path.join(CKPT, f"loss_at_{b // 1000}k.out"))
                # the copy happens at the first poll after the boundary, so the
                # real generation is >= b; record it instead of pretending
                with open(os.path.join(CKPT, "capture_log.jsonl"), "a") as fh:
                    fh.write(json.dumps(dict(tag=tag, generation=g,
                                             boundary=b,
                                             time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
                log(f"captured {tag} at loss.out generation {g}")
        # evaluate anything captured but not yet evaluated
        for b in BOUNDS + [110000]:
            tag = f"gen_{b // 1000}k"
            f = os.path.join(CKPT, tag + ".nep.txt")
            if os.path.exists(f) and not os.path.exists(os.path.join(MS.SER, tag + ".json")):
                if TRIES.get(tag, 0) >= 3:
                    continue
                TRIES[tag] = TRIES.get(tag, 0) + 1
                log(f"battery start {tag} (attempt {TRIES[tag]})")
                try:
                    MS.run_one(tag, f)
                except Exception as e:
                    # log the whole traceback: a one-line message is what made the
                    # first version of this loop silently retry a broken checkpoint
                    import traceback
                    log(f"battery FAILED {tag}: {e}")
                    with open(LOG, "a") as fh:
                        fh.write(traceback.format_exc() + "\n")
                    if TRIES[tag] >= 3:
                        tgt = f + ".failed"
                        n = 1
                        while os.path.exists(tgt):
                            tgt = f + f".failed{n}"
                            n += 1
                        try:
                            os.rename(f, tgt)
                            log(f"giving up on {tag}, moved to {os.path.basename(tgt)}")
                        except OSError:
                            pass
                    time.sleep(A.poll)
                    continue
                log(f"battery done {tag}")
                # refresh the comparison figures after every checkpoint, so the
                # plots are never stale even if the session ends early
                for script in ("make_series_figures.py", "series_report.py"):
                    subprocess.run([sys.executable,
                                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 script)])
                log("figures + report refreshed")
        done = all(os.path.exists(os.path.join(MS.SER, f"gen_{b//1000}k.json"))
                   for b in BOUNDS)
        if done:
            log("all checkpoints done, building figures")
            subprocess.run([sys.executable,
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "make_series_figures.py")])
            log("figures built")
            break
        time.sleep(A.poll)


if __name__ == "__main__":
    main()
