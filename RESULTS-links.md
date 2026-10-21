# Links and fragmentation: reference results (2026-10-21)

AFV: the published Scroll 1 `fibers.afv` (931,104 fibers). Method: `python afv_audit.py --afv <url> --out <dir> --links`
(the `--links` stage). The full run below covered all 375 in-band stretches; a 20-stretch run of the packaged tool
reproduced those 20 stretches exactly (pieces, chains and coverage identical), and two independent full passes agreed on
all 219 stretches they both computed.

## Link precision (dense 2025 cubes)

| | links | correct | share |
|---|---|---|---|
| both ends assigned to a hand fiber | 138 | 110 | 79.7% |
| null: random assigned partner, same cube | 138 | 2-4 (depends on the random draw) | 1.4-2.9% |

Per cube: 32/36, 27/33, 18/29, 19/26, 14/14. Join geometry does not separate wrong links from correct ones (join-angle
AUROC 0.66, 95% CI 0.54-0.76; endpoint-gap AUROC 0.55, 0.42-0.69; pre-registered bar 0.70).

## Fragmentation along held-out human fibers (z 45,000-48,500)

| | value |
|---|---|
| stretches | 375 (374 with an AFV fiber nearby); 2,977,176 voxels of human fiber |
| share of each stretch within 20 voxels of any AFV fiber | median 0.975 (quartiles 0.903-1.000) |
| AFV pieces to reach 90% of that, per stretch | median 2 (quartiles 1-4) |
| linked chains to reach it, per stretch | median 1 (quartiles 1-2) |
| pooled, per 1,000 voxels of human fiber | 0.45 pieces -> 0.22 chains (1,330 -> 653, -51%) |
| stretches covered by a single piece / chain | 47% / 62% |

Caveats: the chain count can be slightly optimistic (a wrong link can pull in a neighbouring fiber's piece within 20 voxels);
link precision is measured outside the held-out band, only where both ends land inside a cube (n = 138).
