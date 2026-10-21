# How good is the published Paris 4 fiber volume? A held-out check of `fibers.afv` against human fibers

Drafted with AI assistance (Claude); reviewed before publishing.

## What this is
An independent check of the published automated fiber volume for Scroll 1 (`PHercParis4/representations/spiral/20260411134726-spiral-dataset-20261019084150/fibers.afv`, 931,104 fibers from the neural fiber follower `polar_tokens_a3_ms_slim_dagger_b32x2_50k/ckpt_050000.pt`, traced over z 28,000-68,000 base voxels) against two sets of human fibers the follower did not learn from.

## What it is not
Not a new tracer, and not a validated filter. The team already audits AFV labels against manual annotations (fibers README, "brought labels from a median 1.46 to 0.68 trace voxels from the manual annotations"). This adds two things that audit does not report publicly: recall on a held-out band, and precision against dense, independent hand tracing. Whether the team has measured either internally is UNCHECKED.

## Recall: does the AFV find the human fibers? (held-out band)
- Judge: the 645 published `eval_fibers`, restricted to z 45,000-48,500 base voxels. Every fiber-follower config in the repo holds this band out of training (`data/fibers.py` `training_state_allowed`, `train/dagger/collect.py`); `ckpt_050000` is a fixed step, not a best-on-validation pick.
- 375 in-band stretches (314 fibers), each scored on a window of up to 600 voxels: an AFV fiber within 20 voxels for 369 of 375.
- Coverage, median (quartiles): single best AFV fiber 0.94 (0.51-1.00) within 10 voxels, 1.00 (0.80-1.00) within 20; all nearby fibers 0.96 within 10, 1.00 within 20. The same stretches shifted 200 voxels: 0.03 within 10, 0.07-0.12 within 20.
- Out-of-band stretches score the same or slightly lower (single 0.86 within 10), so there is no sign the follower memorised the human fibers.

## Precision: are the AFV's fibers real? (dense 2025 cubes)
- Judge: the five Scroll 1 cubes of the 2025 `fiber-skeletons` dataset (WEBKNOSSOS; every fiber in each 256^3 cube traced, 550 fibers). Mapped into the 2026 frame with the volume's own `transform.json`; checked: CT correlation 0.847 between the cube image and the 2026 scan sampled through the transform, ~0 when shifted 30 voxels.
- 1,052 AFV windows (200 base voxels) inside the cubes. Within 20 voxels of a hand fiber for >= 80% of their length (GOOD): 71.5%. Within 20 voxels for <= 20% of their length (BAD): 17.4%. Null (hand fibers shifted 200 voxels): GOOD 4.6%, BAD 69.6%.
- The AFV-to-hand offset is small (per-cube median <= 5 base voxels, spread ~3, about the hand labels' 7.91 um grid), so the 20-voxel tolerance is not hiding a registration error.
- Not an edge effect: BAD and GOOD windows sit at the same distance from the cube faces (median 29.6 vs 30.0 old voxels over all 1,052 windows).
- Per cube, BAD ranges from 7.1% to 29.4% (cube 08997z: 7.1%; cube 00497z_01497y: 29.4%), so one number hides real variation.
- **Open question, stated plainly:** BAD windows are either AFV errors or fibers the 2025 annotators did not trace. A blind visual check at 9.6 um and 2.4 um could not separate a single fiber from the sheet it lies on, so this report does not claim either. Weak hint: BAD windows have lower tube contrast in the CT.

## Links: does following the AFV's own `links` rebuild long fibers? (both judges)

Each AFV fiber stores `fiber_follow.links`, the fibers it continues into at each end.

**Link precision (dense 2025 cubes).** Each AFV fiber inside a cube is assigned the hand fiber it follows (majority vote over its points within 20 voxels of a hand fiber; at least 30 such points; 519 fibers assigned). Of the 138 links whose two ends are both assigned, **110 join two pieces of the same hand fiber (79.7%)**; with a random partner from the same cube instead, 2 of 138 (1.4%) in the exploratory run and 4 of 138 (2.9%) in the packaged tool's random draw. Per cube: 32/36, 27/33, 18/29, 19/26, 14/14. So about one link in five joins two different fibers. Caveat: only links whose two ends are both assigned inside a cube can be scored, so this is a small sample (138 links).

Can join geometry spot the wrong links? Tested with a bar set in advance (AUROC >= 0.70, bootstrap CI above 0.5), no tuning: the angle between the two fibers' end directions gives AUROC 0.66 (95% CI 0.54-0.76) and the endpoint gap 0.55 (0.42-0.69). Wrong links look like correct ones (median join angle 3.9 vs 2.2 degrees; median gap about 47 voxels for both), so a simple geometric filter is not offered.

**Fragmentation (held-out band).** For each in-band stretch of the published human fibers, `union20` is the share within 20 voxels of any AFV fiber, and the count is the fewest AFV pieces (greedy) reaching 90% of that; then the same count after merging pieces connected through `links`.

Over 375 in-band stretches of the published human fibers (median length 4,588 voxels; 2,977,176 voxels in total), the AFV as a whole covers a median **0.97** of each stretch within 20 voxels (quartiles 0.90-1.00); this whole-stretch figure is stricter than the 600-voxel recall windows above.

| | AFV pieces | linked chains |
|---|---|---|
| needed per stretch, median | 2 (quartiles 1-4) | 1 (quartiles 1-2) |
| per 1,000 voxels of human fiber, pooled | 0.45 | 0.22 |
| stretches covered by a single one | 47% | 62% |

Following `links` cuts the number of pieces needed along a held-out human fiber by **51%** (1330 pieces -> 653 chains). 1 stretch with no AFV fiber nearby is left out of the counts.

Two caveats. The chain count can be slightly optimistic: a wrong link that pulls in a neighbouring fiber's piece lying within 20 voxels of the human stretch still counts toward covering it, and fibers on the same sheet can be that close. And link precision comes from the cubes, which lie outside the held-out band (their CT was seen in training, their labels were not).

Reading: following links stitches the AFV's short pieces into fewer, longer chains along real fibers, at the price of the ~20% wrong joins measured above; anyone building long same-winding constraints from the AFV should weigh both numbers.

## A possible filter (in-sample, not validated)
BAD windows sit on short AFV fibers. Dropping fibers shorter than L, on these same windows (thresholds picked after looking):

| L (base voxels) | BAD removed | GOOD removed | BAD share after (before 18.1%) |
|---|---|---|---|
| 1,000 | 27.9% | 10.0% | 15.1% |
| 2,000 | 37.9% | 12.0% | 13.5% |
| 3,000 | 47.1% | 15.7% | 12.2% |

Two learned scorers (window CT + geometry features; then plus fiber metadata) were tested leave-one-cube-out against pre-registered bars and did not clear them (AUROC 0.637 and 0.599 vs 0.645 needed). They are reported, not used.

## Reproduce
`python afv_audit.py --afv <fibers.afv URL> --out ./audit` (see README.md) reproduces every number above; the reference run's
summary is RESULTS.md (precision identical; recall on all 375 in-band stretches). The two learned-scorer tests and the
length table came from the exploratory analysis and are reported for completeness, not packaged.

## Limitations
Five cubes, all from Scroll 1; window labels depend on a 20-voxel tolerance; recall measured on <= 600-voxel windows; the follower's exact run config is inferred from every shipped config, not read from its log.
