# Benchmark artifacts — bf16 saved-state codec (Strata PR #2247)

Raw data and the harness behind the numbers in `docs/STATE_SNAPSHOT_BF16.md` and the state-codec section of
`docs/DETAILS.md` on [PR #2247](https://github.com/Niko1221/Strata/pull/2247), published so every figure can be
re-derived rather than taken on trust.

**Start at [MANIFEST.md](MANIFEST.md).** It maps each file to the doc claim it backs, records which commit each run was
taken on, and lists what is stale.

**Machine, for every run here.** Tesla V100-PCIE-32GB + RTX 2080 Ti, 93 GB RAM (24-core / 32-thread Intel), Linux,
CUDA 12.8, engine `0.1.43.1`, model `qwen3.8-flash-next-iq3_s` IQ3_S, `--kv int8`, `--max-context 262144`, session
files on a Samsung SSD 990 EVO Plus 4TB.

## Layout

| dir | contents |
| --- | --- |
| [results/](results/) | 33 raw JSON files, one per measurement phase and arm. Most records carry their own `commit`, `engine_version`, `codec_under_test`, `engine_args` (read from `/proc/<pid>/cmdline`) and `hardware_static` |
| [harness/](harness/) | the 11 scripts that produced them, plus its README. Deliberately **not** committed to the Strata repo — see [harness/README.md](harness/README.md) |
| [logs/](logs/) | engine stderr for the runs, including `strata-trace.log`, which carries the park byte-by-part trace |

## Headline numbers, on `79955461`

Matched A/B: fresh engine per arm, two configs differing **only** in `--session-state-codec`, `VmRSS` polled at 20 ms,
median of 9, 32K and 63K — `results/memory-{bf16,fp32}.head.json`.

| | 32K | 63K |
| --- | ---: | ---: |
| first `SAVE` after a prefill, `fp32` / `bf16` | +353.2 / **+180.5** MiB | +354.0 / **+181.2** MiB |
| later `SAVE` (allocator reuses the scratch) | ~0, or negative | ~0, or negative |
| `RESTORE`, `fp32` / `bf16` | 350.3 / **406.5** MiB | 805.1 / **861.3** MiB |

The `RESTORE` penalty is **+56.2 MiB at both sizes** — a constant, which is the mechanism: widening allocates the fp32
array the engine runs on beside the coded original still held in the cache, so `bf16` pays live fp32 (112.6 MiB) plus
its own coded state (56.3 MiB) where `fp32` pays the live array alone. The delta equals the coded `bf16` state,
59,031,600 B, to 0.1 MiB.

Parked steady RSS at 32K, both arms: `bf16` 54,325.6 → 56,527.0 MiB (+2,201.4) against `fp32` 54,673.3 → 57,314.4
(+2,641.1), with the cache's own accounting 3,077,756,360 B against 3,727,103,432 B over 4 parked images.

Parked resume, `fp32` against `bf16` median of 2 round-medians (`results/lazy_resume_head_parity.json`): 32K continuing
49.25 / 51.48 ms (**+2.2**), 63K 71.15 / 77.05 (**+5.9**), 121K 121.97 / 127.53 (**+5.6**). On the two code-word
workloads that actually ask for the word, recall was 20/20 in all six cells of both arms.

## Read before quoting anything

- **`SAVE` is two operations, not one.** The first save after a prefill admits a fixed amount of RSS and every later one
  reuses it. A median over repetitions mixes the two — that is how an earlier revision of the doc got a figure that no
  artifact could reproduce. See the *RAM* section of MANIFEST.md.
- **`parked= bytes=` is allocator capacity, not payload.** A fresh park measures 1.00× payload; once the K/V-reuse path
  engages, 2.99×. Pre-existing behaviour from `7f7f8868` and `92a3d5fd`, untouched by this PR, but it means the
  budget-side benefit is smaller than raw byte deltas suggest.
- **Five files carry no commit** (`ab_convert*`, `cold_restore*`) — produced by a harness version that did not record
  provenance. Do not attribute them to a commit.
- **Not measured, and still open:** the KL / top-1-flip curve across tokens-since-resume (not obtainable on this build,
  `/v1/chat/completions` ignores `logprobs`), so the 2⁻⁸ / 0.39 % figure is a *representation* bound and not an
  output-quality result. Also open: HIP (unbuilt, no `hipcc` here), SYCL (unbuilt), and a chain tens of checkpoints deep.

## Scrub

This tree is scrubbed. Every absolute home path on the test box is shown as `<USER>` (1,522 lines across 53 files) and its private LAN address as `<LAN>` (5 files); the hostname appears nowhere. Verified rather than assumed: the whole tree was hashed with those tokens normalised **before** and **after** the rewrite and the digests match, so only identifier tokens changed and every byte of measured payload is exactly as recorded.
tokens normalised **before and after** the rewrite and the digests match, so only identifier tokens changed and every
byte of measured payload is exactly as recorded.