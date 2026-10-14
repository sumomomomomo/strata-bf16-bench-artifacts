"""One engine binary against the other, in one session, on one conversation.

`--session-state-codec` is read once at start-up and so is the engine executable, so the two builds cannot share a
process. This alternates whole server sessions: start an arm, prefill the same text into two conversations, measure
SAVE / RESTORE and a park / resume switch, stop, switch arms. Rounds alternate which arm goes first.

Both arms run with `--adapt-swaps 0 --pcie-frac 0`, which `docs/DETAILS.md:108` names as the way to get repeatable
runs on this box. That is what makes the byte comparison meaningful - without it the two runs' fp32 states differ
before any codec sees them, by about 26x the codec's own error - and it puts both arms on the same footing for time.

    python3 bench_ab_convert.py --tokens 62500 --short 31250 --reps 10 --rounds 2
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile, time

PORT = "8091"
os.environ.setdefault("STRATA_URL", f"http://127.0.0.1:{PORT}")
os.environ.setdefault("STRATA_LOG", "bench-ab-convert.log")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B

ROOT = B.ROOT
ARMS = [("pre-change", os.path.join(ROOT, "engine-cuda12", "strata.pre-change")),
        ("transcode", os.path.join(ROOT, "engine-cuda12", "strata"))]
BASE = os.path.join(ROOT, "bench-state-codec", "bench-bf16.json")   # --base picks the codec under test (fp32 or bf16)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EXTRA = ["--adapt-swaps", "0", "--pcie-frac", "0"]


def engine_running():
    return subprocess.run(["pgrep", "-f", "engine-cuda12/strata"], capture_output=True).returncode == 0


def start(exe, base):
    cfg = json.load(open(base))
    cfg["exe"] = exe
    cfg["args"] = cfg["args"] + EXTRA
    cfg["port"] = int(PORT)
    cfg["log"] = os.path.join(ROOT, os.environ["STRATA_LOG"])
    path = os.path.join(tempfile.gettempdir(), f"ab_convert_{os.path.basename(exe)}.json")
    open(path, "w").write(json.dumps(cfg, indent=1))
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "serve", "server.py"), "--engine", "strata",
                             "--config", path, "--port", PORT], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    B.wait_ready()
    return proc


def stop(proc):
    proc.terminate()
    for _ in range(60):
        if not engine_running():
            break
        time.sleep(1)
    if engine_running():
        subprocess.run(["pkill", "-f", "engine-cuda12/strata"], capture_output=True)
        time.sleep(3)


def measure(doc_long, doc_short, arm, reps):
    """Prefill both conversations, then SAVE / RESTORE and a park / resume switch, `reps` times."""
    out = {"arm": arm, "engine": dict(zip([a for a, _ in ARMS], [e for _, e in ARMS]))[arm]}
    a = B.chat([{"role": "user", "content": doc_long}], 8)
    b = B.chat([{"role": "user", "content": doc_short}], 8)
    out["tokens"] = {"long": a["prompt_tokens"], "short": b["prompt_tokens"]}
    time.sleep(1)

    name = f"ab-{arm}.bin"
    saves, restores = [], []
    for _ in range(reps):
        s = B.save(name)
        saves.append({"engine_ms": s["timings"]["save_ms"], "bytes": s["n_written"],
                      "client_wall_s": round(s["client_wall_s"], 4)})
        r = B.restore(name)
        restores.append({"engine_ms": r["timings"]["restore_ms"], "bytes": r["n_read"],
                         "client_wall_s": round(r["client_wall_s"], 4)})
    out["save_ms"] = B.stats([x["engine_ms"] for x in saves])
    out["restore_ms"] = B.stats([x["engine_ms"] for x in restores])
    out["saves"], out["restores"] = saves, restores
    out["file_bytes"] = os.path.getsize(os.path.join(B.SESSION_DIR, name))
    out["file_sha"] = subprocess.run(["sha256sum", os.path.join(B.SESSION_DIR, name)], capture_output=True,
                                     text=True).stdout.split()[0]

    switches, parks, resumes = [], [], []
    hist = {"A": [{"role": "user", "content": doc_long}, {"role": "assistant", "content": a["answer"]}],
            "B": [{"role": "user", "content": doc_short}, {"role": "assistant", "content": b["answer"]}]}
    for i in range(reps * 2):
        tag = "A" if i % 2 == 0 else "B"
        pos = B.log_bytes()
        r = B.chat(hist[tag] + [{"role": "user", "content": "And one short sentence about it."}], 16)
        lines = B.log_lines_since(pos)
        hist[tag] += [{"role": "user", "content": "And one short sentence about it."},
                      {"role": "assistant", "content": r["answer"]}]
        switches.append({"switch": i, "into": tag, "prompt_ms": r["prompt_ms"], "read": r["prompt_read"],
                         "resume": B.resume_events(lines), "park": B.park_events(lines)})
        parks += [(x["park_ms"], x["image_bytes"]) for x in B.park_events(lines)]
        resumes += [x["resume_ms"] for x in B.resume_events(lines)]
    out["switches"] = switches
    out["park_ms"] = B.stats([x[0] for x in parks])
    out["resume_ms"] = B.stats(resumes)
    out["park_image_bytes"] = sorted({x[1] for x in parks})
    out["ttft_prompt_ms"] = B.stats([x["prompt_ms"] for x in switches])
    out["cache"] = B.cache_counters()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=62500)
    ap.add_argument("--short", type=int, default=31250)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="ab_convert.json")
    a = ap.parse_args()

    doc_long = B.conversation("ABLONG", a.tokens, B.WORDS[3])
    doc_short = B.conversation("ABSHORT", a.short, B.WORDS[4])
    base = a.base
    out = {"phase": "ab_convert", "env": {"base_config": base, "extra_args": EXTRA, "port": PORT,
                                          "arms": dict(ARMS), "log": os.environ["STRATA_LOG"]}, "rounds": []}
    for rnd in range(a.rounds):
        order = ARMS if rnd % 2 == 0 else ARMS[::-1]
        rec = {"round": rnd, "order": [n for n, _ in order], "arms": {}}
        for name, exe in order:
            proc = start(exe, base)
            try:
                rec["arms"][name] = measure(doc_long, doc_short, name, a.reps)
                print(f"round {rnd} {name}: save {rec['arms'][name]['save_ms']['median']} ms  "
                      f"restore {rec['arms'][name]['restore_ms']['median']} ms  "
                      f"resume {rec['arms'][name]['resume_ms']['median']} ms  "
                      f"park {rec['arms'][name]['park_ms']['median']} ms  "
                      f"file {rec['arms'][name]['file_bytes']:,} B  sha {rec['arms'][name]['file_sha'][:12]}",
                      flush=True)
            finally:
                stop(proc)
        out["rounds"].append(rec)
        files = {n: os.path.join(B.SESSION_DIR, f"ab-{n}.bin") for n, _ in ARMS}
        same = subprocess.run(["cmp"] + list(files.values()), capture_output=True)
        rec["files_identical"] = same.returncode == 0
        rec["files_differ_at"] = same.stderr.decode()[:200] if same.returncode else None
        print(f"round {rnd} the two builds' session files: "
              f"{'identical' if rec['files_identical'] else 'differ: ' + (rec['files_differ_at'] or '')}", flush=True)
    B.dump(os.path.join(RESULTS, a.out), out)


main()