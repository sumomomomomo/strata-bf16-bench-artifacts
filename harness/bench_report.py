"""Turns the raw JSON in results/ into the tables for the PR description.

    python3 bench_report.py > results/REPORT.md

Derived rows are labelled: the running-state and K/V split of a parked image is not printed by the engine, so it is
worked out from the two arms' own byte counts (the only difference between them is how the state arrays are held) and
cross-checked against the model geometry. Everything else is read straight out of the JSON.
"""
from __future__ import annotations
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
STATE_FP32 = 118063152          # gdn 117,669,888 + ple 368,640 + tails 18,432 + dead 6,144 + block_pos 48
STATE_BF16 = 59031600           # the same arrays at bf16, block_pos unchanged (int32 positions)


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def md_row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def miB(b):
    return f"{b / 2**20:,.1f} MiB" if b < 2**30 else f"{b / 2**30:.3f} GiB"


def mib_of(b):
    return round(b / 2**20, 1)


def main():
    fp32_sr, bf16_sr = load("save_restore-fp32.json"), load("save_restore-bf16.json")
    fp32_pr, bf16_pr = load("park_resume-fp32.json"), load("park_resume-bf16.json")
    fp32_mem, bf16_mem = load("memory-fp32.json"), load("memory-bf16.json")
    fp32_cold, bf16_cold = load("cold_restore-fp32.json"), load("cold_restore-bf16.json")
    env = (fp32_sr or bf16_sr or {}).get("env", {})

    print("# bf16 saved-state benchmark\n")
    print(f"Commit `{env.get('commit','?')[:10]}`, engine {env.get('engine_version')}, model "
          f"`{env.get('model')}`, K/V `{env.get('kv')}`, context {env.get('max_context')}. "
          f"{env.get('hardware_static',{}).get('cpu_name')} ({env.get('hardware_static',{}).get('cores')} cores), "
          f"{env.get('hardware_static',{}).get('gpu_name')}. One codec per engine process; every other argument, "
          f"prompt and file identical between the arms.\n")

    # ---- session files
    print("## Session SAVE / RESTORE (warm page cache, median of 10 after 2 warm-ups)\n")
    print(md_row(["tokens", "file fp32", "file bf16", "saved", "%", "checkpoint states",
                  "save ms fp32", "save ms bf16", "restore ms fp32", "restore ms bf16"]))
    print(md_row(["---"] * 10))
    for a, b in zip(fp32_sr["sizes"], bf16_sr["sizes"]):
        assert a["actual_tokens"] == b["actual_tokens"], (a["actual_tokens"], b["actual_tokens"])
        saved = a["save_file_bytes"] - b["save_file_bytes"]
        states = round(saved / (STATE_FP32 - STATE_BF16))
        print(md_row([a["actual_tokens"], f"{a['save_file_bytes']:,}", f"{b['save_file_bytes']:,}",
                      f"{saved:,} ({miB(saved)})", f"-{round(100*saved/a['save_file_bytes'],1)}%",
                      f"{states} (live + 1 checkpoint) x {mib_of(STATE_FP32-STATE_BF16)} MiB",
                      f"{a['save_engine_ms']['median']} ({a['save_engine_ms']['min']}-{a['save_engine_ms']['max']})",
                      f"{b['save_engine_ms']['median']} ({b['save_engine_ms']['min']}-{b['save_engine_ms']['max']})",
                      f"{a['restore_engine_ms']['median']} ({a['restore_engine_ms']['min']}-{a['restore_engine_ms']['max']})",
                      f"{b['restore_engine_ms']['median']} ({b['restore_engine_ms']['min']}-{b['restore_engine_ms']['max']})"]))
    print(f"\nFile format versions written: fp32 arm "
          f"{[s.get('save_file_version') for s in fp32_sr['sizes']]}, bf16 arm "
          f"{[s.get('save_file_version') for s in bf16_sr['sizes']]}.\n")

    # ---- cold
    print("### Cold page cache (separate experiment: `posix_fadvise(DONTNEED)` on that one file, then one restore)\n")
    print(md_row(["tokens", "file bf16", "cold restore ms fp32", "cold restore ms bf16", "warm-again ms fp32",
                  "warm-again ms bf16"]))
    print(md_row(["---"] * 6))
    for a, b in zip(fp32_cold["sizes"], bf16_cold["sizes"]):
        print(md_row([a["target_tokens"], f"{b['file_bytes']:,}",
                      f"{a['cold_engine_ms']['median']} ({a['cold_engine_ms']['min']}-{a['cold_engine_ms']['max']})",
                      f"{b['cold_engine_ms']['median']} ({b['cold_engine_ms']['min']}-{b['cold_engine_ms']['max']})",
                      f"{a['warm_engine_ms']['median']}", f"{b['warm_engine_ms']['median']}"]))

    # ---- parked images
    print("\n## Parked conversation images (median of 10 measured switches after 2 warm-ups)\n")
    print(md_row(["tokens", "image fp32", "image bf16", "saved", "%", "states in the image",
                  "park ms fp32/bf16", "resume ms fp32/bf16", "TTFT prompt_ms fp32/bf16", "switches right"]))
    print(md_row(["---"] * 10))
    for a, b in zip(fp32_pr["sizes"], bf16_pr["sizes"]):
        ia, ib = a["image_bytes_parked"][0], b["image_bytes_parked"][0]
        saved = ia - ib
        states = round(saved / (STATE_FP32 - STATE_BF16))
        print(md_row([a["prefill_A"]["tokens"], f"{ia:,} ({miB(ia)})", f"{ib:,} ({miB(ib)})",
                      f"{saved:,} ({miB(saved)})", f"-{round(100*saved/ia,1)}%",
                      f"{states} (live + {states-1} checkpoint chain)",
                      f"{a['park_ms']['median']} / {b['park_ms']['median']}",
                      f"{a['resume_ms']['median']} / {b['resume_ms']['median']}",
                      f"{a['ttft_prompt_ms']['median']} / {b['ttft_prompt_ms']['median']}",
                      f"{a['switches_ok']}/{a['switches_total']} and {b['switches_ok']}/{b['switches_total']}"]))
    print("\nCoded-state count per image, from the byte delta divided by one state's saving "
          f"({STATE_FP32-STATE_BF16:,} B): a parked image holds its live state plus the whole retained checkpoint "
          "chain, each with its own state arrays (`conversation_snapshot_save`: "
          "`conversation_checkpoints_store(view.checkpoints, ...)` then `conversation_checkpoint_save(captured.live, "
          "...)`). The chain reaches its `--prompt-cache 6` cap at 121,615 tokens, which is where the saving stops "
          "growing. The engine's prompt log prints a smaller `checkpoints:` number; that field counts the "
          "checkpoints a request consulted, not the chain a park stores.\n")
    print(md_row(["tokens", "image fp32 (GiB)", "image bf16 (GiB)", "images per 16,384 MiB budget fp32/bf16",
                  "state share of the image fp32/bf16"]))
    print(md_row(["---"] * 5))
    for a, b in zip(fp32_pr["sizes"], bf16_pr["sizes"]):
        ia, ib = a["image_bytes_parked"][0], b["image_bytes_parked"][0]
        n = round((ia - ib) / (STATE_FP32 - STATE_BF16))
        sa, sb = n * STATE_FP32 / ia, n * STATE_BF16 / ib
        print(md_row([a["prefill_A"]["tokens"], f"{ia/2**30:.3f}", f"{ib/2**30:.3f}",
                      f"{16384*2**20/ia:.2f} / {16384*2**20/ib:.2f}", f"{sa*100:.1f}% / {sb*100:.1f}%"]))
    print()

    # ---- memory
    print("## Memory (RSS sampled at 20 ms from /proc on a fresh engine process; median of 3)\n")
    print(md_row(["tokens", "op", "fp32 increment MiB", "bf16 increment MiB", "fp32 engine ms", "bf16 engine ms"]))
    print(md_row(["---"] * 6))
    for a, b in zip(fp32_mem["sizes"], bf16_mem["sizes"]):
        for op in ("save", "restore", "switch_to_A", "switch_to_B"):
            k = f"{op}_increment_mib"
            if k not in a:
                continue
            print(md_row([a["target_tokens"], op, f"{a[k]['median']} ({a[k]['min']}-{a[k]['max']})",
                          f"{b[k]['median']} ({b[k]['min']}-{b[k]['max']})",
                          (a.get(f'{op}_engine_ms') or {}).get('median'), (b.get(f'{op}_engine_ms') or {}).get('median')]))
    print(f"\nSteady process RSS: fp32 " + ", ".join(f"{s['target_tokens']}:{s['steady_rss_mib']:.0f} MiB" for s in fp32_mem["sizes"]) +
          "; bf16 " + ", ".join(f"{s['target_tokens']}:{s['steady_rss_mib']:.0f} MiB" for s in bf16_mem["sizes"]) + ".")
    for arm, d in (("fp32", fp32_mem), ("bf16", bf16_mem)):
        last = d["sizes"][-1]["cache_bytes_after"]
        print(f"{arm}: conversation cache accounted {last['bytes']:,} B over {last['parked']} parked images, "
              f"{last['parks']} parks, {last['restores']} restores, {last['evictions']} evictions.")

    # ---- functional
    print("\n## Functional checks\n")
    print(md_row(["check", "fp32", "bf16"]))
    print(md_row(["---"] * 3))
    for arm in ("fp32", "bf16"):
        pass
    rows = {}
    for arm in ("fp32", "bf16"):
        f, rp, nd = load(f"functional-{arm}.json"), load(f"resume_paths-{arm}.json"), load("needle-125000-%s.json" % arm)
        disk = f["checks"]["needle_across_disk_restore"]
        parked = f["checks"]["needle_across_parked_resume"]
        rows.setdefault("needle 32K across a disk restore (depths 10/50/90 x 3 seeds)",
                        []).append(f"{sum(1 for x in disk if x['restored_ok'])}/{len(disk)}")
        rows.setdefault("needle 32K across a parked resume (same 9)", []).append(
            f"{sum(1 for x in parked if x['ok'])}/{len(parked)}")
        rows.setdefault("needle 121K across a disk restore (3 depths x 1 seed)", []).append(
            f"{sum(1 for x in nd['runs'] if x['restored_ok'])}/{len(nd['runs'])}")
        old = rp["checks"]["older_checkpoint_resume"]
        rows.setdefault("resume from an older retained checkpoint (2 edits)", []).append(
            f"{sum(1 for x in old if x['ok'])}/{len(old)} "
            f"(reused {','.join(str(x['reused']) for x in old)})")
        cyc = rp["checks"]["park_resume_generate_cycles"]
        rows.setdefault("park -> resume -> generate cycles (12)", []).append(
            f"{sum(1 for x in cyc if x['ok'])}/{len(cyc)}")
    for k, v in rows.items():
        print(md_row([k, v[0], v[1]]))

    # ---- ab_engine
    ab = os.path.join(R, "ab_engine.jsonl")
    if os.path.exists(ab):
        rows = [json.loads(l) for l in open(ab)]
        same = {}
        for r in rows:
            same.setdefault((r["round"], r["req"]), {})[r["arm"]] = r["text"]
        pairs = [(k, v) for k, v in same.items() if len(v) == 2]
        print(f"\n`tools/ab_engine.py`, 2 rounds, arms = codec fp32 vs codec bf16: identical answer text in "
              f"{sum(1 for _, v in pairs if v['A'] == v['B'])}/{len(pairs)} paired requests; "
              f"decode medians within "
              f"{max(abs((x['decode'] or 0) - (y['decode'] or 0)) for (_, _), x, y in [(k, next(r for r in rows if r['arm']=='A' and r['req']==k[1] and r['round']==k[0]), next(r for r in rows if r['arm']=='B' and r['req']==k[1] and r['round']==k[0])) for k, _ in pairs]):.1f} tok/s.")

    # ---- expected vs measured
    print("\n## Expected against measured\n")
    print(f"One running state at this geometry: {STATE_FP32:,} B fp32, {STATE_BF16:,} B coded, "
          f"saving {STATE_FP32-STATE_BF16:,} B ({miB(STATE_FP32-STATE_BF16)}).")
    for a, b in zip(fp32_sr["sizes"], bf16_sr["sizes"]):
        n = round((a["save_file_bytes"] - b["save_file_bytes"]) / (STATE_FP32 - STATE_BF16))
        print(f"- session file at {a['actual_tokens']:,} tokens: {n} states stored x {STATE_FP32-STATE_BF16:,} B = "
              f"{n*(STATE_FP32-STATE_BF16):,} B expected, {a['save_file_bytes']-b['save_file_bytes']:,} B measured.")
    for a, b in zip(fp32_pr["sizes"], bf16_pr["sizes"]):
        ia, ib = a["image_bytes_parked"][0], b["image_bytes_parked"][0]
        n = round((ia - ib) / (STATE_FP32 - STATE_BF16))
        print(f"- parked image at {a['prefill_A']['tokens']:,} tokens: {n} states x {STATE_FP32-STATE_BF16:,} B = "
              f"{n*(STATE_FP32-STATE_BF16):,} B expected, {ia-ib:,} B measured.")


main()