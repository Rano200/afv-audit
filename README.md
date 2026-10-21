# afv_audit

Checks a Paris 4 / Scroll 1 fiber volume (`fibers.afv`, the output of the neural fiber follower) against human-traced fibers.
One script, plain Python, Linux / macOS / Windows.

## What it measures

1. **Precision against dense hand tracing.** The five Scroll 1 cubes (256^3) of the 2025 `fiber-skeletons` dataset have every
   fiber traced by hand (550 fibers). The tool downloads their WEBKNOSSOS NMLs from `dl.ash2txt.org`, maps the nodes into the
   2026 base-voxel frame with the 2026 volume's `transform.json` (matrix inverted), cuts every AFV fiber inside each cube into
   non-overlapping 200-voxel windows (a window must lie inside the cube, checked in the 2023 frame), and labels each window by
   the share of its points within `tol` base voxels of any hand fiber: **GOOD** >= 0.8, **BAD** <= 0.2.
   Null: the hand fibers shifted 200 voxels in XY (angle seeded by the cube's z); the same windows are re-labelled against them.
2. **Recall on a held-out band.** The published `eval_fibers` (645 human fibers) restricted to z in `--recall-band`
   (default 45000-48500, the band the follower was trained without). Each in-band stretch (>= 200 voxels of arc length) is scored
   on a deterministic random window of up to 600 voxels: the share within `tol` of the best single AFV fiber, and of the union
   of all AFV fibers. Null: the same window shifted 200 voxels in XY.
3. Two cheap sanity checks on (1): the per-cube AFV-to-hand offset (median and MAD of the vector to the nearest hand point), and
   the median distance of GOOD vs BAD windows to the cube faces (detects an edge artifact).

4. **Optional, `--links` (off by default; existing outputs unchanged): links and fragmentation.**
   - *Link precision.* Inside the five dense cubes each AFV fiber is assigned the hand fiber it follows (majority vote over its
     in-cube points within 20 voxels of a hand fiber, at least 30 such points). A link `f -> g` from `fiber_follow.links` with both
     ends assigned is CORRECT if both follow the same hand fiber. Null: `g` replaced by a random assigned fiber in the same cube (rng seed 7).
   - *Fragmentation.* For each held-out in-band stretch: `union20` = share within 20 voxels of ANY AFV fiber; `pieces` = greedy
     minimum number of AFV fibers reaching 0.9 x union20; `chains` = the same after merging fibers joined by `fiber_follow.links`
     (union-find over the candidate fibers). Reported as medians/quartiles, per 1,000 voxels, pooled reduction %, and the share of
     single-piece / single-chain stretches. Results go in the "Links and fragmentation" section of `summary.md` / `summary.json`;
     per-stretch rows are appended to `frag.jsonl` (resumable). `--links-sample N` scores N random in-band stretches (seed 12345).

Constants, seeds and window rules are those of the analysis behind REPORT.md, unchanged; RESULTS.md is the reference run's summary.

## What it does NOT claim

- **BAD windows are not proven AFV errors.** A BAD window is an AFV fiber with no hand fiber nearby. It is either a wrong AFV
  fiber or a real fiber the 2025 annotators did not trace (an annotator omission). The tool cannot tell which. Read BAD as
  "unconfirmed by hand tracing", not "false".
- Agreement is not correctness: if the AFV and the humans share a systematic error, it will not show.
- GOOD/BAD labels depend on the tolerance (20 voxels is much looser than 10); read both.
- Recall windows are capped at 600 voxels, so whole-fiber recall is lower than the window numbers. Recall is "did the AFV find the
  human fiber", not "is every AFV fiber real".
- Only five cubes, all Scroll 1. The held-out band is held out of the follower's training per its configs; that is not re-verified here.
- `--links`: link precision is measured only where BOTH linked fibers fall inside a cube and have an assignment (small n, five
  Scroll 1 cubes); it says nothing about links elsewhere. A wrong link merges different fibers, so the chain count and the pooled
  reduction can be flattered by merge errors; fewer chains is not proof the chains are real single fibers. Greedy cover is not an exact minimum.
- The tests use synthetic numbers for mechanics only and prove nothing about any AFV.

## Run

```
pip install numpy scipy requests apsw        # apsw is needed only to read an AFV over HTTP
python afv_audit.py --afv https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/representations/spiral/20260411134726-spiral-dataset-20261019084150/fibers.afv --out ./audit
```

Options: `--recall-band ZLO ZHI`, `--cubes all` (or comma-separated name substrings), `--tol 10 20`,
`--recall-sample N` (score N random in-band stretches instead of all), `--workers 4`, `--skip-recall`, `--skip-precision`, `--links`, `--links-sample N`.
A local `.afv` path works with the standard-library `sqlite3` (no apsw; the file is 11 GB). The `rtree` SQLite module must be present
(it is in standard CPython builds and in apsw).

An AFV is a SQLite file. Over HTTP the tool reads it with range requests through an apsw virtual file system (64 KiB chunks).
Everything downloaded (AFV chunks, NMLs, transform, eval_fibers) is cached under `<out>/cache` and `<out>/afv_chunks`;
re-running skips finished stages, and the recall stage appends to `recall.jsonl` and resumes where it stopped. Delete `<out>` to start clean.

## Runtime (reference run, 4 workers, home connection, empty cache)

Precision about 1.5 minutes (~300 MB of AFV chunks). Recall on all 375 in-band stretches: about 9 minutes (4 workers, empty cache; ~150 MB of eval_fibers JSON plus AFV chunks). Total ~11 minutes.
RAM: under 1 GB per worker.

## Outputs (in `--out`)

- `summary.md`, `summary.json` - the tables (precision per tolerance and per cube, offsets, edge check; recall median / quartiles).
- `cube_fibers_2026.json` - hand fibers in the 2026 frame. `precision_windows.json` - one row per AFV window with its near-fractions.
- `recall.jsonl` - one row per scored stretch. `frag.jsonl` - one row per fragmentation stretch (`--links` only).

## Tests

`python -m unittest discover -s tests` - checks the transform inversion and the window labelling on a three-point example by hand arithmetic, and the greedy cover / union-find mechanics on synthetic lines.

## Known differences from the draft report

The draft report's edge-check figures (median 34 vs 38 voxels) came from the feature-table windows (869 windows). This tool computes
the edge check on the 1,052 precision windows' midpoints, so those two numbers differ slightly; the precision table is identical.
