#!/usr/bin/env python3
"""afv_audit.py - audit a Vesuvius Challenge fiber volume (.afv) against human-traced fibers.

    python afv_audit.py --afv <url-or-path to fibers.afv> --out <dir>
                        [--recall-band 45000 48500] [--cubes all] [--tol 10 20]
                        [--recall-sample N] [--workers 4]

Stages (each is cached and resumable under --out):
  1. cube labels : download the five 2025 Scroll 1 fiber-skeleton NMLs (dl.ash2txt.org) and map them into the 2026
                   Paris 4 frame with the 2026 volume's transform.json, inverted.
  2. precision   : cut AFV fibers inside each 256^3 cube into 200-base-voxel windows; label each window by the share of
                   its points within `tol` base voxels of any hand fiber (GOOD >= 0.8, BAD <= 0.2); null = hand fibers
                   shifted 200 voxels in XY (a random angle seeded by the cube's z).
  3. recall      : for the published eval_fibers, take stretches with z in the held-out band, score a <= 600-voxel
                   window of each against the AFV (best single fiber / union of fibers within tol), null = the same
                   window shifted 200 voxels in XY.
  4. summary     : summary.json and summary.md.

Method is a port of fiberscore/{skeleton_labels,cube_labels,cube_offset,confound_edge}.py and
fiberprobe_scripts/{stretches,judge,judge_sum}.py; constants and seeds are unchanged. See README.md.

An AFV is a SQLite file. A local path is read with the stdlib sqlite3. A URL is read by HTTP range requests through an
SQLite virtual file system, which needs the `apsw` package (pip install apsw); 64 KiB chunks are cached on disk.
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

import numpy as np
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------------------------- constants
VOL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/"
       "20260411134726-2.400um-0.2m-78keV-masked.zarr")
NML_BASE = "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset001_sk-fibers-20250124/nml/"
EVAL_BASE = "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/eval_fibers/"
CUBES = ["fibers_s1a_00497z_01497y_03997x_256_v00", "fibers_s1a_00497z_02497y_02997x_256_v00",
         "fibers_s1a_00997z_02497y_02997x_256_v00", "fibers_s1a_08997z_02997y_02497x_256_v00",
         "fibers_s1a_10997z_02997y_02997x_256_v00"]
WIN = 200.0            # precision window length, base voxels
GOOD_T, BAD_T = 0.8, 0.2
RWIN = 600.0           # recall window length (arc length), base voxels
MARGIN = 25.0          # block_bounds query margin
CH = 1 << 16           # AFV range-read chunk
CUBE_RE = re.compile(r"fibers_s1a_(\d+)z_(\d+)y_(\d+)x_(\d+)")


# ----------------------------------------------------------------------------------------------- generic helpers
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def http_get(url, dest=None, tries=6, rng=None):
    """GET with retries. If dest is given the body is cached there (atomic write) and reused if present."""
    if dest and os.path.exists(dest):
        return open(dest, "rb").read()
    err = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            break
        except Exception as e:  # noqa: BLE001 - retry any network error
            err = e
            time.sleep(1 + t)
    else:
        raise RuntimeError("fetch failed %s: %s" % (url, err))
    if dest:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".tmp%d" % os.getpid()
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
    return data


def densify(pts):
    """Linear interpolation so consecutive points are <= 1 voxel apart (same as overlap2.densify)."""
    out = [pts[:1]]
    for u, v in zip(pts[:-1], pts[1:]):
        n = int(np.ceil(np.linalg.norm(v - u)))
        out.append(u + (v - u) * (np.arange(1, n + 1)[:, None] / n) if n > 1 else v[None])
    return np.concatenate(out)


def rseed(*a):
    return int(hashlib.md5("|".join(map(str, a)).encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------------------------- transform + NMLs
def load_transform(cache_dir, vol=VOL):
    """Return (A, Ainv): 4x4 matrix mapping 2026 base voxels (x,y,z) -> 2023 7.91 um frame, and its inverse."""
    raw = http_get(vol + "/transform.json", os.path.join(cache_dir, "transform.json"))
    M = np.array(json.loads(raw)["transformation_matrix"], float)
    A = np.vstack([M, [0, 0, 0, 1]])
    return A, np.linalg.inv(A)


def apply_h(T, p):
    """Apply a 4x4 homogeneous transform to an (N,3) array."""
    p = np.asarray(p, float)
    return (T @ np.c_[p, np.ones(len(p))].T)[:3].T


def _farthest(start, adj):
    dist, prev, q = {start: 0}, {start: None}, deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v], prev[v] = dist[u] + 1, u
                q.append(v)
    end = max(dist, key=dist.get)
    path = []
    while end is not None:
        path.append(end)
        end = prev[end]
    return path


def longest_path(nodes, edges):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    if not edges:
        return list(nodes)
    a = _farthest(next(iter(adj)), adj)[0]
    return _farthest(a, adj)


def build_cube_labels(out, cubes, Ainv):
    """Stage 1. NML -> polylines in the 2026 frame. Cached as cube_fibers_2026.json."""
    dest = os.path.join(out, "cube_fibers_2026.json")
    if os.path.exists(dest):
        have = json.load(open(dest))
        if {c for c in cubes} <= {o["cube"] for o in have}:
            log("stage 1: cached", dest)
            return have
    res = []
    for cube in cubes:
        raw = http_get(NML_BASE + cube + ".nml", os.path.join(out, "cache", "nml", cube + ".nml"))
        root = ET.fromstring(raw)
        for thing in root.iter("thing"):
            pos = {int(n.get("id")): (float(n.get("x")), float(n.get("y")), float(n.get("z"))) for n in thing.iter("node")}
            edges = [(int(e.get("source")), int(e.get("target"))) for e in thing.iter("edge")]
            if len(pos) < 2:
                continue
            order = longest_path(pos, [(a, b) for a, b in edges if a in pos and b in pos])
            p26 = apply_h(Ainv, [pos[i] for i in order])
            res.append(dict(cube=cube, fiber_id=int(thing.get("id")), n_nodes=len(pos), n_path=len(order),
                            points=p26.round(2).tolist()))
    json.dump(res, open(dest, "w"))
    log("stage 1: %d hand fibers" % len(res), dict(collections.Counter(o["cube"] for o in res)))
    return res


# ----------------------------------------------------------------------------------------------- AFV access
_STATE = {}


def _make_http_vfs(url, cache_dir):
    try:
        import apsw
    except ImportError:
        sys.exit("Reading an AFV over HTTP needs the 'apsw' package: pip install apsw\n"
                 "(or download fibers.afv and pass its local path).")
    os.makedirs(cache_dir, exist_ok=True)
    size_f = os.path.join(cache_dir, "size.txt")
    if os.path.exists(size_f):
        size = int(open(size_f).read())
    else:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            size = int(r.headers["Content-Range"].split("/")[1])
        open(size_f, "w").write(str(size))
    mem = collections.OrderedDict()
    stats = {"req": 0, "bytes": 0}

    def chunk(i):
        if i in mem:
            mem.move_to_end(i)
            return mem[i]
        lo, hi = i * CH, min(size, i * CH + CH) - 1
        fn = os.path.join(cache_dir, "%d" % (i // 1000), "%d.bin" % i)
        if os.path.exists(fn):
            data = open(fn, "rb").read()
        else:
            data = http_get(url, fn, rng="bytes=%d-%d" % (lo, hi))
            stats["req"] += 1
            stats["bytes"] += len(data)
        mem[i] = data
        if len(mem) > 600:
            mem.popitem(last=False)
        return data

    class F:
        def __init__(self, name, flags): pass

        def xRead(self, amount, offset):
            out = b""
            while amount > 0:
                i = offset // CH
                c = chunk(i)
                o = offset - i * CH
                piece = c[o:o + amount]
                if not piece:
                    out += b"\0" * amount
                    break
                out += piece
                offset += len(piece)
                amount -= len(piece)
            return out

        def xFileSize(self): return size
        def xFileControl(self, op, ptr): return False
        def xWrite(self, data, offset): raise apsw.ReadOnlyError()
        def xTruncate(self, s): raise apsw.ReadOnlyError()
        def xSync(self, flags): pass
        def xClose(self): pass
        def xLock(self, lvl): pass
        def xUnlock(self, lvl): pass
        def xCheckReservedLock(self): return False
        def xSectorSize(self): return 4096
        def xDeviceCharacteristics(self): return 0x1000 | 0x2000  # immutable-ish

    vname = "afvhttp%d" % os.getpid()

    class V(apsw.VFS):
        def __init__(self): super().__init__(vname, "")
        def xOpen(self, name, flags): return F(name, flags)
        def xFullPathname(self, n): return n
        def xAccess(self, n, f): return True

    vfs = V()
    conn = apsw.Connection("file:/fibers.afv?immutable=1", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI,
                           vfs=vname)
    return conn, vfs, stats


def init_worker(src, cache_dir):
    """Open the AFV in this process (also called in the parent)."""
    _STATE.clear()
    if re.match(r"https?://", src):
        conn, vfs, stats = _make_http_vfs(src, os.path.join(cache_dir, "afv_chunks"))
        _STATE.update(conn=conn, vfs=vfs, stats=stats)
    else:
        _STATE["conn"] = sqlite3.connect("file:%s?mode=ro&immutable=1" % os.path.abspath(src).replace("\\", "/"), uri=True)


def cursor():
    return _STATE["conn"].cursor()


def query_blocks(cur, lo, hi):
    """block ids whose bounds touch the box [lo, hi] (inclusive, as in the original scripts)."""
    return [i for (i,) in cur.execute(
        "select id from block_bounds where min_x<=? and max_x>=? and min_y<=? and max_y>=? and min_z<=? and max_z>=?",
        (float(hi[0]), float(lo[0]), float(hi[1]), float(lo[1]), float(hi[2]), float(lo[2])))]


def fetch_blocks(cur, ids):
    """{fiber_id: [(first_segment, points (n,3))]} for the given block ids."""
    per = collections.defaultdict(list)
    ids = sorted(ids)
    for a in range(0, len(ids), 400):
        sub = ids[a:a + 400]
        for fid, fseg, blob in cur.execute("select fiber_id, first_segment, points from blocks where id in (%s)"
                                           % ",".join(map(str, sub))):
            per[fid].append((fseg, np.frombuffer(blob, dtype="<f8").reshape(-1, 3)))
    return per


# ----------------------------------------------------------------------------------------------- precision
def cut_windows(run, win=WIN):
    """Non-overlapping 200-arc-length windows along a polyline (same rule as cube_labels.windows)."""
    seg = np.linalg.norm(np.diff(run, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    out, a = [], 0.0
    while a + win <= s[-1]:
        m = (s >= a) & (s <= a + win)
        if m.sum() >= 2:
            out.append(run[m])
        a += win
    return out


def label_from_fraction(f):
    return "GOOD" if f >= GOOD_T else "BAD" if f <= BAD_T else "MID"


def cube_geometry(cube, A, Ainv):
    z, y, x, S = map(int, CUBE_RE.match(cube).groups())
    lo23, hi23 = np.array([x, y, z], float), np.array([x + S, y + S, z + S], float)
    corners = np.array([[a, b, c] for a in (lo23[0], hi23[0]) for b in (lo23[1], hi23[1]) for c in (lo23[2], hi23[2])])
    c26 = apply_h(Ainv, corners)
    return z, lo23, hi23, c26.min(0), c26.max(0)


def precision_cube(args):
    cube, fibers_pts, A, Ainv, tols = args
    cur = cursor()
    hand = [densify(np.array(p, float)) for p in fibers_pts]
    z, lo23, hi23, lo, hi = cube_geometry(cube, A, Ainv)
    per = fetch_blocks(cur, query_blocks(cur, lo, hi))
    hp = np.concatenate(hand)
    tree = cKDTree(hp)
    ang = np.random.default_rng(z).uniform(0, 2 * np.pi)
    tree_null = cKDTree(hp + np.array([np.cos(ang), np.sin(ang), 0.0]) * 200.0)
    rows, runs_all = [], []
    for fid, parts in per.items():
        pts = densify(np.concatenate([p for _, p in sorted(parts, key=lambda t: t[0])]))
        p23 = apply_h(A, pts)
        inside = np.all((p23 >= lo23) & (p23 <= hi23), axis=1)
        if inside.sum() < 50:
            continue
        idx = np.flatnonzero(inside)
        for run_idx in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            runs_all.append(pts[run_idx])
            for w in cut_windows(pts[run_idx]):
                d, _ = tree.query(w)
                dn, _ = tree_null.query(w)
                mid23 = apply_h(A, ((w[0] + w[-1]) / 2)[None])[0]
                face = float(min((mid23 - lo23).min(), (hi23 - mid23).min()))
                rows.append(dict(cube=cube, afv_fiber=int(fid), start=w[0].round(1).tolist(), end=w[-1].round(1).tolist(),
                                 face_dist_2023=face,
                                 **{"near%d" % t: float((d <= t).mean()) for t in tols},
                                 **{"null_near%d" % t: float((dn <= t).mean()) for t in tols}))
    # registration residual: vector from AFV points to nearest hand point (<= 25 voxels), median/MAD per cube
    allp = np.concatenate(runs_all) if runs_all else np.zeros((0, 3))
    off = None
    if len(allp):
        d, j = tree.query(allp)
        m = d <= 25
        if m.any():
            vec = hp[j[m]] - allp[m]
            med = np.median(vec, axis=0)
            off = dict(n=int(m.sum()), median_xyz=med.round(2).tolist(),
                       mad_xyz=np.median(np.abs(vec - med), axis=0).round(2).tolist())
    return dict(cube=cube, rows=rows, offset=off, n_afv_fibers=len(per), n_hand=len(hand),
                stats=dict(_STATE.get("stats", {})))


def run_precision(out, src, fibers, cubes, A, Ainv, tols, workers):
    dest = os.path.join(out, "precision_windows.json")
    if os.path.exists(dest):
        d = json.load(open(dest))
        if set(d["cubes"]) >= set(cubes) and d["tols"] == list(tols):
            log("stage 2: cached", dest)
            return d
    by = collections.defaultdict(list)
    for o in fibers:
        by[o["cube"]].append(o["points"])
    jobs = [(c, by[c], A, Ainv, tols) for c in cubes]
    log("stage 2: precision over %d cubes" % len(jobs))
    t0 = time.time()
    with cf.ProcessPoolExecutor(min(workers, len(jobs)), initializer=init_worker, initargs=(src, out)) as ex:
        res = []
        for r in ex.map(precision_cube, jobs):
            log("  %s: AFV fibers %d, windows %d, hand fibers %d (%ds)" % (r["cube"], r["n_afv_fibers"], len(r["rows"]),
                                                                      r["n_hand"], time.time() - t0))
            res.append(r)
    d = dict(cubes=cubes, tols=list(tols), rows=[x for r in res for x in r["rows"]],
             offsets={r["cube"]: r["offset"] for r in res})
    json.dump(d, open(dest, "w"))
    return d


def summarize_precision(d):
    rows = d["rows"]
    out = {"n_windows": len(rows), "per_tol": {}, "per_cube": {}, "offsets": d["offsets"]}
    for t in d["tols"]:
        v = np.array([r["near%d" % t] for r in rows])
        vn = np.array([r["null_near%d" % t] for r in rows])
        out["per_tol"][str(t)] = dict(good=float((v >= GOOD_T).mean()), bad=float((v <= BAD_T).mean()),
                                      null_good=float((vn >= GOOD_T).mean()), null_bad=float((vn <= BAD_T).mean()))
    t = max(d["tols"])
    for c in d["cubes"]:
        v = np.array([r["near%d" % t] for r in rows if r["cube"] == c])
        out["per_cube"][c] = dict(n=int(len(v)), good=float((v >= GOOD_T).mean()) if len(v) else None,
                                  bad=float((v <= BAD_T).mean()) if len(v) else None)
    v = np.array([r["near%d" % t] for r in rows])
    f = np.array([r["face_dist_2023"] for r in rows])
    out["edge_check"] = dict(tol=t, median_face_dist_good=float(np.median(f[v >= GOOD_T])) if (v >= GOOD_T).any() else None,
                             median_face_dist_bad=float(np.median(f[v <= BAD_T])) if (v <= BAD_T).any() else None)
    return out


# ----------------------------------------------------------------------------------------------- recall
def list_eval_files(out):
    html = http_get(EVAL_BASE, os.path.join(out, "cache", "eval_index.html")).decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="([^"/]+\.json)"', html)))


def get_eval(out, name):
    return json.loads(http_get(EVAL_BASE + name, os.path.join(out, "cache", "eval_fibers", name)))


def runs_of(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > 1)
    st = np.concatenate([[0], br + 1])
    en = np.concatenate([br, [len(idx) - 1]])
    return [(int(idx[a]), int(idx[b])) for a, b in zip(st, en)]


def make_stretches(P, band):
    """In-band stretches (z within the band) with arc length >= 200 (stretches.py)."""
    res = []
    for a, b in runs_of((P[:, 2] >= band[0]) & (P[:, 2] <= band[1])):
        if b <= a:
            continue
        L = float(np.linalg.norm(np.diff(P[a:b + 1], axis=0), axis=1).sum())
        if L >= 200:
            res.append((a, b, L))
    return res


def recall_window(P, a, b, file):
    seg = P[a:b + 1]
    cl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
    if cl[-1] <= RWIN:
        return seg
    s0 = np.random.default_rng(rseed(file, a, "in")).uniform(0, cl[-1] - RWIN)
    return seg[(cl >= s0) & (cl <= s0 + RWIN)]


def coverage_vs_afv(cur, R, tols):
    w = np.linalg.norm(np.diff(R, axis=0), axis=1)
    w = np.concatenate([w, [0]])
    w = (w + np.roll(w, 1)) / 2
    bids = set()
    for a in range(0, len(R), 100):
        p = R[a:a + 101]
        bids.update(query_blocks(cur, p.min(0) - MARGIN, p.max(0) + MARGIN))
    per = collections.defaultdict(list)
    for fid, parts in fetch_blocks(cur, bids).items():
        per[fid] = [densify(p) for _, p in parts]
    dl = [(fid, np.concatenate(bl)) for fid, bl in per.items()]
    best = {t: 0.0 for t in tols}
    u = {t: 0.0 for t in tols}
    mind = None
    if dl:
        for _fid, dense in dl:
            dist, _ = cKDTree(dense).query(R)
            for t in tols:
                best[t] = max(best[t], float(w[dist <= t].sum() / w.sum()))
        dist, _ = cKDTree(np.concatenate([d for _, d in dl])).query(R)
        for t in tols:
            u[t] = float(w[dist <= t].sum() / w.sum())
        mind = float(dist.min())
    return dict(best={str(t): v for t, v in best.items()}, union={str(t): v for t, v in u.items()},
                ncand=len(dl), anyw20=bool(mind is not None and mind <= 20), length=float(w.sum()))


def recall_job(args):
    out, file, a, b, tols = args
    P = np.array(get_eval(out, file)["line_points"], float)
    R = recall_window(P, a, b, file)
    rng = np.random.default_rng(rseed(file, a, "shift"))
    ang = rng.uniform(0, 2 * np.pi)
    shift = np.array([np.cos(ang), np.sin(ang), 0.0]) * 200.0
    rec = dict(file=file, a=a, b=b, zmin=float(R[:, 2].min()), zmax=float(R[:, 2].max()))
    try:
        rec["real"] = coverage_vs_afv(cursor(), R, tols)
        rec["shift200"] = coverage_vs_afv(cursor(), R + shift, tols)
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)
    return rec


def run_recall(out, src, band, tols, sample, workers):
    files = list_eval_files(out)
    log("stage 3: %d eval_fibers listed; downloading (cached)" % len(files))
    with cf.ThreadPoolExecutor(8) as ex:
        evs = list(ex.map(lambda f: (f, np.array(get_eval(out, f)["line_points"], float)), files))
    st = [(f, a, b) for f, P in evs for a, b, _ in make_stretches(P, band)]
    log("  in-band stretches: %d from %d fibers" % (len(st), len({s[0] for s in st})))
    if sample and sample < len(st):
        pick = np.random.default_rng(12345).choice(len(st), sample, replace=False)
        st = [st[i] for i in sorted(pick)]
        log("  SAMPLE: %d stretches (rng seed 12345)" % len(st))
    jl = os.path.join(out, "recall.jsonl")
    done = {}
    if os.path.exists(jl):
        for l in open(jl):
            try:
                r = json.loads(l)
                if "error" not in r:
                    done[(r["file"], r["a"], r["b"])] = r
            except ValueError:
                pass
    todo = [(out, f, a, b, tols) for f, a, b in st if (f, a, b) not in done]
    log("  %d done, %d to run" % (len(st) - len(todo), len(todo)))
    if todo:
        t0 = time.time()
        with cf.ProcessPoolExecutor(workers, initializer=init_worker, initargs=(src, out)) as ex, open(jl, "a") as fo:
            for n, r in enumerate(ex.map(recall_job, todo, chunksize=2), 1):
                fo.write(json.dumps(r) + "\n")
                fo.flush()
                if "error" not in r:
                    done[(r["file"], r["a"], r["b"])] = r
                if n % 20 == 0:
                    log("  recall %d/%d (%ds)" % (n, len(todo), time.time() - t0))
    return [done[(f, a, b)] for f, a, b in st if (f, a, b) in done], len(st)


def summarize_recall(rows, tols, band, n_requested):
    def q(x):
        x = np.array(x)
        return dict(median=float(np.median(x)), q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)), mean=float(x.mean()))
    res = {"band": list(band), "n_stretches": len(rows), "n_requested": n_requested,
           "n_fibers": len({r["file"] for r in rows}), "median_window_len": float(np.median([r["real"]["length"] for r in rows])),
           "cond": {}}
    for c in ("real", "shift200"):
        res["cond"][c] = {"any_afv_within_20": int(sum(r[c]["anyw20"] for r in rows)), "tol": {}}
        for t in tols:
            k = str(t)
            res["cond"][c]["tol"][k] = dict(single=q([r[c]["best"][k] for r in rows]), union=q([r[c]["union"][k] for r in rows]),
                                            single_ge_07=int(sum(r[c]["best"][k] >= .7 for r in rows)),
                                            union_ge_07=int(sum(r[c]["union"][k] >= .7 for r in rows)))
    return res


# ----------------------------------------------------------------------------------------------- summary
def pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def write_summary(out, prec, rec, args):
    js = dict(afv=args.afv, precision=prec, recall=rec, tolerances=args.tol)
    json.dump(js, open(os.path.join(out, "summary.json"), "w"), indent=1)
    L = ["# AFV audit summary", "", "AFV: `%s`" % args.afv, ""]
    if prec:
        L += ["## Precision (AFV windows inside 5 dense 2025 hand-traced cubes; %d windows of 200 base voxels)" % prec["n_windows"], "",
              "GOOD = >= 80% of the window within tol of a hand fiber; BAD = <= 20%. Null = hand fibers shifted 200 voxels.", "",
              "| tol (base voxels) | GOOD | BAD | null GOOD | null BAD |", "|---|---|---|---|---|"]
        for t, v in prec["per_tol"].items():
            L.append("| %s | %s | %s | %s | %s |" % (t, pct(v["good"]), pct(v["bad"]), pct(v["null_good"]), pct(v["null_bad"])))
        L += ["", "Per cube (largest tol):", "", "| cube | windows | GOOD | BAD |", "|---|---|---|---|"]
        for c, v in prec["per_cube"].items():
            L.append("| %s | %d | %s | %s |" % (c, v["n"], pct(v["good"]), pct(v["bad"])))
        L += ["", "AFV-to-hand offset (median xyz, MAD xyz; AFV points with a hand point within 25 voxels):", ""]
        for c, v in prec["offsets"].items():
            L.append("- %s: %s" % (c, "none" if v is None else "median %s, MAD %s (n=%d)" % (v["median_xyz"], v["mad_xyz"], v["n"])))
        e = prec["edge_check"]
        L += ["", "Edge check (tol %s): median distance to cube face, 2023 voxels: GOOD %.1f, BAD %.1f." % (
            e["tol"], e["median_face_dist_good"], e["median_face_dist_bad"]), ""]
    if rec:
        L += ["## Recall (eval_fibers stretches with z in %s; %d stretches from %d fibers%s)" % (
            rec["band"], rec["n_stretches"], rec["n_fibers"],
            "" if rec["n_stretches"] >= rec["n_requested"] else " - SAMPLE of %d" % rec["n_requested"]), "",
            "Share of each <= 600-voxel window within tol of the best single AFV fiber / the union of all AFV fibers (median, quartiles).", "",
            "| condition | tol | single median (Q1-Q3) | union median (Q1-Q3) | single >= 0.7 | any AFV within 20 |", "|---|---|---|---|---|---|"]
        for c, cv in rec["cond"].items():
            for t, v in cv["tol"].items():
                s, u = v["single"], v["union"]
                L.append("| %s | %s | %.2f (%.2f-%.2f) | %.2f (%.2f-%.2f) | %d/%d | %d |" % (
                    c, t, s["median"], s["q1"], s["q3"], u["median"], u["q1"], u["q3"], v["single_ge_07"], rec["n_stretches"],
                    cv["any_afv_within_20"]))
        L.append("")
    L += ["## Caveats", "",
          "- BAD windows are AFV fibers with no hand fiber nearby. They may be AFV errors OR fibers the 2025 annotators did not trace; "
          "this audit does not tell which.",
          "- Agreement with human traces is not correctness; a shared systematic error would not show.",
          "- Five cubes, all Scroll 1. Recall windows are capped at 600 voxels, so whole-fiber recall is lower.", ""]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L))


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--afv", required=True, help="URL or local path of fibers.afv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recall-band", nargs=2, type=float, default=[45000, 48500], metavar=("ZLO", "ZHI"))
    ap.add_argument("--cubes", default="all", help="'all' or comma-separated cube names / substrings")
    ap.add_argument("--tol", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--recall-sample", type=int, default=0, help="score only N random in-band stretches (0 = all)")
    ap.add_argument("--skip-recall", action="store_true")
    ap.add_argument("--skip-precision", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    cubes = CUBES if args.cubes == "all" else [c for c in CUBES if any(s in c for s in args.cubes.split(","))]
    if not cubes:
        sys.exit("no cube matches --cubes")
    tols = sorted(set(args.tol))
    t0 = time.time()
    prec = rec = None
    if not args.skip_precision:
        A, Ainv = load_transform(os.path.join(out, "cache"))
        fibers = build_cube_labels(out, cubes, Ainv)
        prec = summarize_precision(run_precision(out, args.afv, fibers, cubes, A, Ainv, tols, args.workers))
    if not args.skip_recall:
        rows, n_req = run_recall(out, args.afv, tuple(args.recall_band), tols, args.recall_sample, args.workers)
        rec = summarize_recall(rows, tols, args.recall_band, n_req) if rows else None
    write_summary(out, prec, rec, args)
    log("done in %.0fs; see %s" % (time.time() - t0, os.path.join(out, "summary.md")))


if __name__ == "__main__":
    main()
