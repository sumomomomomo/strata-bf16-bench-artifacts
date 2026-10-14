"""Cold-page-cache RESTORE, measured separately from the warm numbers.

The warm benchmark restores a file that was written and re-read seconds earlier, so it is entirely in the page
cache. This evicts one file's own pages with posix_fadvise(POSIX_FADV_DONTNEED) - no root, and nothing else in the
system's cache is touched - then restores it once and times that. Each repetition re-evicts, so every repetition is
a cold read of that file. The engine still has to read the bytes off the drive.

    python3 bench_cold.py bf16 31250,62500,125000
"""
from __future__ import annotations
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B

codec = sys.argv[1]
sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "31250,62500,125000").split(",")]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
B.wait_ready()

out = {"phase": "cold_restore", "codec": codec, "method":
       "os.posix_fadvise(fd, POSIX_FADV_DONTNEED) on the one session file, then one restore, timed client-side; "
       "engine restore_ms reported alongside. No root, no global cache drop, no other file's pages touched.",
       "drive": os.popen("lsblk -o MODEL -d /dev/nvme0n1 | tail -1").read().strip(),
       "filesystem": os.popen("df -T " + ROOT + " | tail -1").read().split()[-2], "sizes": []}

for tokens in sizes:
    fname = f"sr-{codec}-{tokens}.bin"
    path = os.path.join(B.SESSION_DIR, fname)
    if not os.path.exists(path):
        out["sizes"].append({"target_tokens": tokens, "error": f"{fname} not written by this arm"})
        continue
    cold, warm = [], []
    for i in range(reps):
        fd = os.open(path, os.O_RDONLY)

        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        r = B.restore(fname)
        cold.append({"engine_restore_ms": r["timings"]["restore_ms"], "client_wall_s": round(r["client_wall_s"], 4),
                     "bytes": r["n_read"]})
        r = B.restore(fname)                       # straight back in: this one is warm again
        warm.append({"engine_restore_ms": r["timings"]["restore_ms"], "client_wall_s": round(r["client_wall_s"], 4),
                     "bytes": r["n_read"]})
    rec = {"target_tokens": tokens, "file": fname, "file_bytes": os.path.getsize(path), "reps": reps,
           "cold": cold, "warm_after_eviction": warm,
           "cold_engine_ms": B.stats([x["engine_restore_ms"] for x in cold]),
           "cold_client_s": B.stats([x["client_wall_s"] for x in cold]),
           "warm_engine_ms": B.stats([x["engine_restore_ms"] for x in warm])}
    out["sizes"].append(rec)
    print(f"{tokens}: file {rec['file_bytes']:,} B  cold median {rec['cold_engine_ms']['median']} ms "
          f"(min {rec['cold_engine_ms']['min']}, max {rec['cold_engine_ms']['max']})  "
          f"warm-again median {rec['warm_engine_ms']['median']} ms", flush=True)

B.dump(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"cold_restore-{codec}.json"), out)