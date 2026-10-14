# Benchmark artifacts — bf16 saved-state codec (PR #2247)

Raw data behind the numbers in `docs/STATE_SNAPSHOT_BF16.md` and the state-codec section of `docs/DETAILS.md`.
Published so every figure can be re-derived rather than taken on trust.

**Machine, for every run here.** Tesla V100-PCIE-32GB + RTX 2080 Ti, 93 GB RAM (24-core / 32-thread Intel), Linux,
CUDA 12.8, model `qwen3.8-flash-next-iq3_s` IQ3_S, `--kv int8`, `--max-context 262144`, engine `0.1.43.1`.
Session files on a Samsung SSD 990 EVO Plus 4TB. Paths and the LAN address are scrubbed (see *Scrub* below); no
measured number was altered.

## How to read this

- `results/*.json` — one file per measurement phase. Field names are the harness's own (`park_ms`, `resume_ms`,
  `engine_prompt_ms`, `retained_bytes`, `image_bytes`, `rss_increment_mib`, …). Most records carry their own
  `commit`, `engine_version`, `codec_under_test`, `engine_args` and `hardware_static`.
- `harness/*.py` — the scripts that produced them. They are **not** committed to the Strata repository, deliberately:
  another session reset `bench-state-codec/` away twice, and the docs now say the harness is local. This bundle is the
  reference the PR review asked for.
- `logs/*.log` — engine stderr for the runs, including `strata-trace.log`, which carries the park byte-by-part trace,
  and `bench-mem-*.log` / `bench-mem-*.server.log` from the RAM re-measurement.
- `--adapt-swaps 0 --pcie-frac 0` is pinned in the lazy/capture A/B phases. The RAM re-measurement is **not** pinned:
  it runs on the plain bench configs so the two arms differ only in `--session-state-codec`. Every record carries
  `engine_args` read from `/proc/<pid>/cmdline`, so this is checkable per run rather than taken on trust. Fresh engine
  per arm. Warm-ups then measured repetitions, counts in each file.

## Provenance

`engine_commit` recorded in a file is the **working-tree HEAD at run time**, which for binaries built from an earlier
commit is not the source that binary was built from. Where the two differ it is called out in the last column and you
should trust the binary name over the commit.

| file | phase | codec | recorded commit | binary | records | backs |
| --- | --- | --- | --- | --- | ---: | --- |
| `lazy_resume_head_parity.json` | lazy/fp32-vs-bf16 A/B | both | `e79abfc0` | `strata.head` | 12 | the 6-row parked-resume table, the gains table, the park by-arm table |
| `capture_codec.json` | capture + turn latency | both | `e79abfc0` | `strata.head` | 8 | *Measured: capture and turn latency, 16384 against 8192* |
| `capture_codec_smoke.json` | smoke test only | bf16 | `e79abfc0` | `strata.head` | 1 | nothing — validation run, kept for completeness |
| `lazy_resume_codec_parity2.json` | earlier fp32-vs-bf16 A/B | both | `387b9c3b` | `strata.parity` | 12 | the "same A/B on the build before capture-time coding" comparison |
| `lazy_resume_codec_parity.json` | partial A/B | both | `bbb167c6` | `strata.lazy2` | 4 | superseded by the two above; retained as history |
| `lazy_resume.json` | eager vs lazy restoration | bf16 | `2274c1c` | `strata.eager-2274c1c`, `strata.lazy` | 12 | *Measured: eager (2274c1c) against lazy, both bf16* |
| `lazy_resume-fp32-default.json` | default-path check | fp32 | `2274c1c4` | `strata.lazy` | 1 | *The default path* — confirms neither new log line prints |
| `memory-{bf16,fp32}.head.json` | RAM re-measurement on HEAD | both | `79955461` | `strata` | 2 sizes each | the RAM section: cold vs warm `SAVE`, the +56.2 MiB `RESTORE` delta, parked RSS and cache accounting |
| `memory-{bf16,fp32}.head.n3.json` | first pass of the same, n=3 | both | `79955461` | `strata` | 2 sizes each | shows the `SAVE` bimodality that made n=3 inadequate |
| `ab_convert.json`, `ab_convert-fp32.json`, `ab_convert-63k-32k.json` | encode/decode A/B | both | **none recorded** | `strata`, `strata.pre-change` | 1 each | the −31.6 / −31.7 ms park pairs |
| `functional-{bf16,fp32}.json` | functional correctness | both | `d178753d` | — | 2 each | flag-on behaviour, codec log line, cache accounting |
| `save_restore-{bf16,fp32}.json`, `save_restore-fp32-oversized.json` | session file bytes, `save_ms`, `restore_ms` | both | `d178753d` | — | 1 each | the 4K / 32K / 63K / 128K session-file table |
| `park_resume-{bf16,fp32}.json`, `park_resume-bf16.pre-change.json` | parked image bytes, `park_ms` | both | `d178753d` | — | 1 each | the parked-image table and images-per-budget |
| `resume_paths-{bf16,fp32}.json`, `resume_paths-bf16.pre-change.json` | restore paths | both | `d178753d` | — | 2 each | resume `ms` per path, the codec-gated log lines |
| `memory-{bf16,fp32}.json`, `memory-bf16.pre-change.json`, `memory-bf16.transcode.json`, `memory-fp32-cache-full.json` | `VmRSS` during SAVE / RESTORE | both | `d178753d` | — | 1 each | the RAM section, +353.2 / +240.8 / +415.2 MiB |
| `needle-125000-{bf16,fp32}.json` | needle recall at 121K | both | `d178753d` | — | 3 each | recall after save/restore at depth |
| `cold_restore-{bf16,fp32}.json` | cold (page-cache-dropped) restore | both | **none recorded** | — | — | the cold-read column of the parked-image section |

`d178753d` is *state codec: a stored array converts to another codec in one pass* — the pre-lazy, pre-capture-coding
codec build. `2274c1c` predates lazy restoration entirely.

## What is stale, stated plainly

1. **`lazy_resume.json` (the eager-vs-lazy table) predates `3f04cc72`**, which made a retained checkpoint take the
   configured codec at capture instead of at park. Its comparison of two *restoration* builds still holds, because
   `3f04cc72` applies to both sides equally — but its retained-byte, `VmRSS` and `park` columns are the
   pre-capture-coding case.
2. **The 63K row of the eager-vs-lazy table is one round for both arms.** 32K and 121K are medians of two rounds.
3. **In `capture_codec.json` every cell reports `ok=0/10`.** The turn prompt asks the model to name the code word
   *without quoting the document* within 24 tokens, so the check fails by construction. Latency in that file is valid;
   correctness in that file is not measured. Recall comes from `lazy_resume_head_parity.json`, where the two workloads
   that do ask for the word scored 20/20 in all six cells of both arms.
4. **`parked= bytes=` and `snapshot_bytes` are allocator capacity, not payload.** `ConversationBuffer::bytes()` sums
   `segment.capacity()`. A fresh park measures 1.00× payload; after the K/V-reuse path engages, 2.99× — see
   `logs/strata-trace.log`, parks 1, 3 and 5 (184.7 MiB payload against 184.7 / 554.1 / 554.1 MiB allocated). This is
   pre-existing park behaviour from `7f7f8868` and `92a3d5fd`, untouched by this PR, but it means the budget-side
   benefit of bf16 is smaller than the raw byte deltas suggest. Do not quote parked-bytes deltas as budget-side savings
   without re-reading that trace.
5. **The `park` figures in the committed doc were restated in `79955461`** as within-arm, across-build comparisons on
   two-round medians. An earlier revision quoted single rounds (`78.85` / `89.2`) — those are superseded.
6. **The five files marked "none recorded" carry no commit.** `ab_convert*` and `cold_restore*` were produced by a
   harness version that did not record provenance. Their binaries are named in the table; treat their machine and flags
   as the same box and pinning, but do not attribute them to a commit I cannot verify.

## The four unreproducible RAM figures — re-measured, and what they turned out to be

The audit that assembled the first version of this bundle found four claims in the RAM section of
`docs/STATE_SNAPSHOT_BF16.md` that no surviving artifact supported: the bf16 `RESTORE` +415.2 MiB at 32K, `SAVE`
+353.8/+240.8 at 63K, `RESTORE` +805 at 63K, and the parked steady-RSS/cache pair 57,342 / 56,897 MiB over 3 images.
All four were re-measured on `79955461` — fresh engine per arm, `bench-bf16.json` against `bench-fp32.json` differing
only in `--session-state-codec`, `VmRSS` polled at 20 ms, median of 9, 32K and 63K — in
`memory-bf16.head.json` and `memory-fp32.head.json`. The n=3 pass that came first is kept as `*.head.n3.json`.

| arm | size | first save after prefill | later saves (median of 9) | `RESTORE` (median of 9, spread) | steady RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| `bf16` | 32K | **+180.5** | −66.7 (max 158.6) | **406.5** (390.9–406.6) | 54,325.6 |
| `bf16` | 63K | **+181.2** | −490.2 (max 0.1) | **861.3** (861.0–861.3) | 58,616.9 |
| `fp32` | 32K | **+353.2** | −1.5 (max 15.6) | **350.3** (334.6–350.5) | 54,673.3 |
| `fp32` | 63K | **+354.0** | −392.5 (max −0.2) | **805.1** (789.1–805.1) | 59,907.9 |

**`SAVE` was never one operation.** The first save after a prefill admits a fixed amount of RSS; every later save reuses
it and reads ~0, or negative at 63K as page cache is written back while the image goes to disk. The old `+353.2 /
+240.8` pair came from a median of 3 that mixed the two cases — the per-rep values in `memory-fp32.head.n3.json` are
353.2, −107.2, −12.2 — and the surviving `memory-fp32.json` reports median = min = max = 353.2, three identical values,
which is what a caught-once-and-reused allocation looks like when three reps are collapsed into one statistic. The bf16
half of the old pair lived in a file the harness keyed on codec alone, and a second size overwrote it. The admission is
not token-sized (32K and 63K agree to 1 MiB) and is far larger than the 4 MiB `kStateCodecChunkFloats` coding scratch;
it scales with the codec at 0.51×. No further attribution is claimed — this PR does not instrument the allocator.

**The bf16 `RESTORE` claim was right in direction and is now pinned.** +56.2 MiB at 32K and +56.2 MiB at 63K — a
constant, which is the mechanism: widening allocates the fp32 array the engine runs on beside the coded original still
held in the cache, so bf16 pays live fp32 (112.6 MiB) plus its own coded state (56.3 MiB) where fp32 pays the live array
alone. The delta equals the coded bf16 state, 59,031,600 B = 56.30 MiB, to 0.1 MiB. The old +415.2 is not reproduced and
appears in no artifact as median or max; the closest is `memory-bf16.transcode.json`'s 3-sample **max** of 419.2.

**Parked steady RSS, restated with the count actually reached** — 4 parked images, not 3, because the A and B
conversations from the size's own windows are still resident: `bf16` 54,325.6 → 56,527.0 MiB (+2,201.4) with the cache
accounting 3,077,756,360 B, against `fp32` 54,673.3 → 57,314.4 MiB (+2,641.1) with 3,727,103,432 B. The old 57,342 /
56,897 figures were close to these but described a 3-image state nothing here records.

Two caveats the doc keeps: `VmRSS` misses pages the kernel already wrote out and most parked state is pageable, so every
increment here is an observed lower bound; and once the conversation cache is at its budget the increments read zero
because the allocator already owns the blocks, which is why both arms run on a fresh process.

## Not measured — and still open

- The KL / top-1-flip curve across tokens-since-resume. **Not obtainable on this build**: `/v1/chat/completions`
  ignores `logprobs`. This is the measurement that decides whether bf16 state is safe as a default, and it is the
  reason the 2^-8 (0.39 %) figure in the docs is a *representation* bound and not an output-quality result.
- Retained checkpoint bytes before the first park.
- A memory-tight conversation that used to refuse and now fits.
- A chain tens of checkpoints deep, where the spacing term dominates.
- Capture during a prefill rather than between turns.
- HIP: unbuilt, no `hipcc` on this box. `conversation_state.cpp` and `generate.cpp` are shared with it.
- SYCL: unbuilt, no oneAPI. `sycl/src/program/generate.cpp` changed without a compile.
- The layer-split park/restore path is covered by unit tests only; every engine run here is single-GPU.

## Scrub

This tree is scrubbed. Every absolute home path on the test box is shown as `<USER>` (1,522 lines across 53 files) and its private LAN address as `<LAN>` (5 files); the hostname appears nowhere. Verified rather than assumed: the whole tree was hashed with those tokens normalised **before** and **after** the rewrite and the digests match, so only identifier tokens changed and every byte of measured payload is exactly as recorded.

## Excluded

`strata-iq3_s.log` — the live server log of the running instance. Unrelated traffic, and it would only add noise.