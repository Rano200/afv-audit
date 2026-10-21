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
