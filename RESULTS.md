# AFV audit summary

AFV: `https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/representations/spiral/20260411134726-spiral-dataset-20261019084150/fibers.afv`

## Precision (AFV windows inside 5 dense 2025 hand-traced cubes; 1052 windows of 200 base voxels)

GOOD = >= 80% of the window within tol of a hand fiber; BAD = <= 20%. Null = hand fibers shifted 200 voxels.

| tol (base voxels) | GOOD | BAD | null GOOD | null BAD |
|---|---|---|---|---|
| 10 | 34.8% | 37.9% | 0.3% | 92.5% |
| 20 | 71.5% | 17.4% | 4.6% | 69.6% |

Per cube (largest tol):

| cube | windows | GOOD | BAD |
|---|---|---|---|
| fibers_s1a_00497z_01497y_03997x_256_v00 | 228 | 60.5% | 29.4% |
| fibers_s1a_00497z_02497y_02997x_256_v00 | 235 | 70.6% | 17.0% |
| fibers_s1a_00997z_02497y_02997x_256_v00 | 218 | 66.1% | 18.3% |
| fibers_s1a_08997z_02997y_02497x_256_v00 | 210 | 84.8% | 7.1% |
| fibers_s1a_10997z_02997y_02997x_256_v00 | 161 | 78.3% | 13.0% |

AFV-to-hand offset (median xyz, MAD xyz; AFV points with a hand point within 25 voxels):

- fibers_s1a_00497z_01497y_03997x_256_v00: median [-1.55, 1.61, -1.86], MAD [3.01, 3.14, 3.44] (n=50405)
- fibers_s1a_00497z_02497y_02997x_256_v00: median [1.69, 4.74, 0.41], MAD [2.48, 2.92, 3.13] (n=59439)
- fibers_s1a_00997z_02497y_02997x_256_v00: median [2.17, 4.22, -1.42], MAD [3.03, 3.22, 3.29] (n=52873)
- fibers_s1a_08997z_02997y_02497x_256_v00: median [-2.98, 0.39, 0.89], MAD [2.57, 3.02, 2.69] (n=57830)
- fibers_s1a_10997z_02997y_02997x_256_v00: median [-0.86, -0.28, 0.12], MAD [1.93, 2.97, 2.35] (n=41437)

Edge check (tol 20): median distance to cube face, 2023 voxels: GOOD 30.0, BAD 29.6.

## Recall (eval_fibers stretches with z in [45000, 48500]; 375 stretches from 314 fibers)

Share of each <= 600-voxel window within tol of the best single AFV fiber / the union of all AFV fibers (median, quartiles).

| condition | tol | single median (Q1-Q3) | union median (Q1-Q3) | single >= 0.7 | any AFV within 20 |
|---|---|---|---|---|---|
| real | 10 | 0.94 (0.51-1.00) | 0.96 (0.70-1.00) | 252/375 | 369 |
| real | 20 | 1.00 (0.80-1.00) | 1.00 (1.00-1.00) | 288/375 | 369 |
| shift200 | 10 | 0.03 (0.00-0.04) | 0.03 (0.00-0.05) | 0/375 | 334 |
| shift200 | 20 | 0.07 (0.05-0.11) | 0.12 (0.05-0.21) | 2/375 | 334 |

## Caveats

- BAD windows are AFV fibers with no hand fiber nearby. They may be AFV errors OR fibers the 2025 annotators did not trace; this audit does not tell which.
- Agreement with human traces is not correctness; a shared systematic error would not show.
- Five cubes, all Scroll 1. Recall windows are capped at 600 voxels, so whole-fiber recall is lower.
