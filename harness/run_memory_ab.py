#!/usr/bin/env python3
"""Run the RAM re-measurement as a matched A/B on the current build, then put the user's server back.

Starts a server from bench-bf16.json, runs bench_memory_rerun.py for both sizes in one invocation, stops it, repeats
from bench-fp32.json, and finally restarts the server from strata-iq3_s.json. The two bench configs differ only in the
value of --session-state-codec, so the arms are matched. No --adapt-swaps/--pcie-frac pinning is added: the original
memory runs did not use it, and env_block records the real argv from /proc so this is checkable after the fact.

Anything that fails still restores the server, because the box is serving someone's conversation.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import bench_lib as B
import bench_lazy_resume as L                                  # engine_running / port_open only

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(B.ROOT, "engine-cuda12", "strata")          # the HEAD binary
CONFIGS = {"bf16": os.path.join(HERE, "bench-bf16.json"), "fp32": os.path.join(HERE, "bench-fp32.json")}
PORT = os.environ.get("MEM_PORT", "8080")
USER_CONFIG = os.path.join(B.ROOT, "strata-iq3_s.json")
SIZES = sys.argv[1:] or ["31250", "62500"]


def kill_server():
    """Wrapper and engine, by pid. Never `pkill -f engine-cuda12/strata`: that matches this script's own argv."""
    subprocess.run(["pkill", "-f", "serve/server.py --engine strata"], capture_output=True)
    pid = B.engine_pid()
    if pid:
        subprocess.run(["kill", str(pid)], capture_output=True)
    for _ in range(180):
        if not L.engine_running() and not L.port_open():
            time.sleep(3)                                       # let the kernel take the locked experts back
            return True
        time.sleep(1)
    return False


def start(cfg_path, log_name):
    cfg = json.load(open(cfg_path))
    cfg["exe"] = EXE
    cfg["port"] = int(PORT)
    cfg["log"] = os.path.join(B.ROOT, log_name)
    path = os.path.join(tempfile.gettempdir(), f"mem_{os.path.basename(cfg_path)}")
    open(path, "w").write(json.dumps(cfg, indent=1))
    server_log = os.path.join(B.ROOT, f"bench-mem-{os.path.basename(cfg_path).replace('.json', '')}.server.log")
    out = open(server_log, "w")
    proc = subprocess.Popen([sys.executable, os.path.join(B.ROOT, "serve", "server.py"), "--engine", "strata",
                             "--config", path, "--port", PORT], cwd=B.ROOT, stdout=out, stderr=subprocess.STDOUT)
    end = time.time() + 900
    while time.time() < end:
        if proc.poll() is not None:
            tail = open(server_log, errors="replace").read()[-1500:]
            raise SystemExit(f"server exited ({proc.returncode}) before ready. Last of {server_log}:\n{tail}")
        try:
            B.get("/health", 5)
            return proc
        except Exception:
            time.sleep(2)
    raise SystemExit("server did not come up in 900 s")


def stop(proc):
    proc.terminate()
    for _ in range(120):
        if not L.engine_running():
            break
        time.sleep(1)
    pid = B.engine_pid()
    if pid:
        subprocess.run(["kill", str(pid)], capture_output=True)
    for _ in range(180):
        if proc.poll() is not None and not L.engine_running() and not L.port_open():
            break
        time.sleep(1)
    time.sleep(5)


def restore_user_server():
    if not kill_server():
        print("WARN: could not free the port for the user's server", flush=True)
        return
    out = open(os.path.join(B.ROOT, "restore-user-server.log"), "w")
    subprocess.Popen([sys.executable, os.path.join(B.ROOT, "serve", "server.py"), "--engine", "strata",
                      "--config", USER_CONFIG, "--port", PORT], cwd=B.ROOT, stdout=out, stderr=subprocess.STDOUT)
    end = time.time() + 900
    while time.time() < end:
        try:
            B.get("/health", 5)
            print("user server restored on port", PORT, flush=True)
            return
        except Exception:
            time.sleep(2)
    print("WARN: user server did not report healthy within 900 s", flush=True)


def main():
    print("HEAD:", os.popen(f"git -C {B.ROOT} rev-parse --short HEAD").read().strip(),
          "| sizes:", SIZES, "| binary:", EXE, flush=True)
    kill_server()
    try:
        for codec in ("bf16", "fp32"):
            print(f"\n=== arm {codec}: starting server ===", flush=True)
            proc = start(CONFIGS[codec], f"bench-mem-{codec}.log")
            m = B.get("/metrics")
            print(f"  engine {m['engine'].get('version')} | codec line check via log | model {m['engine'].get('model')}",
                  flush=True)
            env = dict(os.environ, STRATA_URL=f"http://127.0.0.1:{PORT}", STRATA_LOG=f"bench-mem-{codec}.log")
            r = subprocess.run([sys.executable, os.path.join(HERE, "bench_memory_rerun.py"), codec, *SIZES],
                               cwd=HERE, env=env)
            print(f"  arm {codec}: harness exit {r.returncode}", flush=True)
            stop(proc)
    finally:
        restore_user_server()


if __name__ == "__main__":
    main()