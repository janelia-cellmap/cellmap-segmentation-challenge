import numpy as np
from scipy import ndimage as ndi


def perturb_mask_iou_3d(
    G,
    target_iou,
    band=2,
    p_fn=0.5,
    rng=None,
    empty_k=64,  # how many voxels to turn on if G is empty
    empty_blob_radius=1,  # blob radius in voxels (1 => 3x3x3-ish ball)
    max_tries=4000,
):
    """
    3D spatially-realistic perturbation:
      P = G, then remove FN mainly from inner boundary band, add FP mainly to outer boundary band.
    Attempts to achieve IoU ~= target_iou; if exact feasibility is tricky, chooses the closest feasible pair.

    Graceful behavior:
      - If G is empty: return a small random blob of positives (size ~ empty_k).
    """
    G = np.asarray(G, dtype=bool)
    if rng is None:
        rng = np.random.default_rng()

    t = float(target_iou)
    if not (0 < t <= 1):
        raise ValueError("target_iou must be in (0, 1].")

    # ---- Empty GT: fail gracefully by returning some predicted positives ----
    g = int(G.sum())
    if g == 0:
        P = np.zeros_like(G, dtype=bool)
        # make a small spherical-ish blob at a random center
        zdim, ydim, xdim = P.shape
        cz = rng.integers(0, zdim)
        cy = rng.integers(0, ydim)
        cx = rng.integers(0, xdim)

        r = int(max(0, empty_blob_radius))
        zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
        se = (xx * xx + yy * yy + zz * zz) <= r * r
        coords = np.array(np.nonzero(se)).T - r  # offsets

        # place offsets, clipping to volume
        pts = []
        for dz, dy, dx in coords:
            z, y, x = cz + dz, cy + dy, cx + dx
            if 0 <= z < zdim and 0 <= y < ydim and 0 <= x < xdim:
                pts.append((z, y, x))

        # if blob is too small/large relative to empty_k, sprinkle random voxels too
        for z, y, x in pts:
            P[z, y, x] = True

        need = max(0, int(empty_k) - int(P.sum()))
        if need > 0:
            flat = rng.choice(P.size, size=min(need, P.size), replace=False)
            P.flat[flat] = True
        return P

    if t == 1:
        return G.copy()

    bg = ~G
    b = int(bg.sum())

    def iou_from_ra(r, a):
        # For this construction: |G∩P| = g-r, |G∪P| = g+a
        return (g - r) / (g + a) if (g + a) > 0 else 0.0

    def a_from_r(r):
        # integer FP count induced by r to hit target (subject to rounding)
        return int(round((g - r) / t - g))

    # ---- Choose an initial r near expected error split ----
    # total “error mass” ~ g*(1-t); allocate p_fn to FN portion heuristically
    r0 = int(np.clip(round(p_fn * g * (1 - t)), 0, g))
    jitter = max(5, g // 80)

    best = None  # (abs_error, r, a, achieved_iou)
    found_exactish = False

    # ---- 1) Local randomized search around r0 (fast path) ----
    for _ in range(int(max_tries)):
        r = int(np.clip(r0 + rng.integers(-jitter, jitter + 1), 0, g))
        a = a_from_r(r)
        if 0 <= a <= b:
            achieved = iou_from_ra(r, a)
            err = abs(achieved - t)
            cand = (err, r, a, achieved)
            if best is None or cand < best:
                best = cand
            # If we're extremely close, stop early.
            if err <= (1.0 / max(1, g)):  # “one-voxel” scale tolerance
                found_exactish = True
                break

    # ---- 2) Guaranteed fallback: broaden search deterministically if needed ----
    if best is None or (not found_exactish and best[0] > 1.0 / max(1, g)):
        # Try a coarse sweep of r values (bounded cost) to guarantee feasibility.
        # We avoid O(g) when g is huge by sampling r candidates across [0,g].
        n = 2048  # cap candidates
        if g <= n:
            r_candidates = np.arange(g + 1, dtype=int)
        else:
            r_candidates = np.unique(
                np.concatenate(
                    [
                        np.linspace(0, g, n, dtype=int),
                        np.array(
                            [max(0, r0 - 4 * jitter), r0, min(g, r0 + 4 * jitter)],
                            dtype=int,
                        ),
                    ]
                )
            )

        for r in r_candidates:
            a = a_from_r(int(r))
            if 0 <= a <= b:
                achieved = iou_from_ra(int(r), int(a))
                err = abs(achieved - t)
                cand = (err, int(r), int(a), achieved)
                if best is None or cand < best:
                    best = cand

    if best is None:
        # Absolute last resort: clamp a into feasible range and pick r to minimize error.
        # This guarantees we return something rather than crash.
        r = int(np.clip(r0, 0, g))
        a = int(np.clip(a_from_r(r), 0, b))
        best = (abs(iou_from_ra(r, a) - t), r, a, iou_from_ra(r, a))

    _, r, a, achieved_iou = best

    # ---- Build boundary bands for spatial realism ----
    band = int(band)
    if band > 0:
        zz, yy, xx = np.ogrid[-band : band + 1, -band : band + 1, -band : band + 1]
        se = (xx * xx + yy * yy + zz * zz) <= band * band
        inner = G & ~ndi.binary_erosion(G, structure=se)
        outer = ndi.binary_dilation(G, structure=se) & ~G
    else:
        inner, outer = G, ~G

    P = G.copy()

    def sample(mask, k):
        idx = np.flatnonzero(mask)
        if k <= 0 or idx.size == 0:
            return np.array([], dtype=np.int64)
        k = min(int(k), idx.size)
        return rng.choice(idx, size=k, replace=False)

    # FN removals (prefer inner band)
    drop = sample(inner, r)
    P.flat[drop] = False
    rem = r - drop.size
    if rem > 0:
        drop2 = sample(P & G, rem)
        P.flat[drop2] = False

    # FP additions (prefer outer band)
    add = sample(outer & ~P, a)
    P.flat[add] = True
    rem = a - add.size
    if rem > 0:
        add2 = sample((~G) & ~P, rem)
        P.flat[add2] = True

    return P

# ----------------- normalization -----------------

def normalize_distance(distance: float, voxel_size) -> float:
    if distance == np.inf:
        return 0.0
    return float(1.01 ** (-distance / np.linalg.norm(voxel_size)))


def inv_normalize_distance(score: float, voxel_size) -> float:
    """Inverse of normalize_distance for score in (0,1]. Returns physical distance."""
    score = float(score)
    if score <= 0.0:
        return np.inf
    if score >= 1.0:
        return 0.0
    return -np.linalg.norm(voxel_size) * (np.log(score) / np.log(1.01))


# ----------------- geometry helpers -----------------

def _surface_3d(M: np.ndarray) -> np.ndarray:
    st = ndi.generate_binary_structure(3, 1)  # 6-connect
    return M & ~ndi.binary_erosion(M, structure=st, iterations=1, border_value=0)


def _ball_se(r: int) -> np.ndarray:
    r = int(r)
    zz, yy, xx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    return (xx*xx + yy*yy + zz*zz) <= r*r


def _edge_margin_mm(shape, voxel_size) -> np.ndarray:
    """Per-voxel minimum physical distance to array boundary."""
    Z, Y, X = shape
    vz, vy, vx = voxel_size
    z = np.arange(Z)[:, None, None]
    y = np.arange(Y)[None, :, None]
    x = np.arange(X)[None, None, :]
    dz = np.minimum(z, (Z - 1) - z) * vz
    dy = np.minimum(y, (Y - 1) - y) * vy
    dx = np.minimum(x, (X - 1) - x) * vx
    return np.minimum(np.minimum(dz, dy), dx)


def _pick_seed_on_surface_max_margin(S: np.ndarray, voxel_size, rng: np.random.Generator) -> int | None:
    """Pick a surface voxel flat-index, preferring max boundary margin (stable for 'out' bumps)."""
    sidx = np.flatnonzero(S)
    if sidx.size == 0:
        return None
    margin = _edge_margin_mm(S.shape, voxel_size).ravel()[sidx]
    mmax = margin.max()
    top = sidx[margin >= (mmax - 1e-6)]
    return int(rng.choice(top))


def _max_radius_out_mm(M: np.ndarray, voxel_size) -> float:
    """Max outward bump radius limited by distance to volume boundary."""
    S = _surface_3d(M)
    if not S.any():
        return 0.0
    margin = _edge_margin_mm(M.shape, voxel_size)
    return float(margin[S].max()) * 0.95


def _max_radius_in_mm(M: np.ndarray, voxel_size) -> float:
    """Max inward dent radius limited by object thickness."""
    if not M.any():
        return 0.0
    dt_in = ndi.distance_transform_edt(M, sampling=voxel_size)
    return float(dt_in.max()) * 0.95


# ----------------- hausdorff on voxels (your definition, no ROI needed for report) -----------------

def hausdorff_voxels_mm(A: np.ndarray, B: np.ndarray, voxel_size) -> float:
    """
    Symmetric Hausdorff distance between full voxel sets A and B (physical units).
    Mirrors your "standard" method (max of directed maxima).
    """
    A = A.astype(bool)
    B = B.astype(bool)
    a_n = int(A.sum())
    b_n = int(B.sum())
    if a_n == 0 and b_n == 0:
        return 0.0
    if a_n == 0 or b_n == 0:
        return np.inf

    dt_to_B = ndi.distance_transform_edt(~B, sampling=voxel_size)
    dt_to_A = ndi.distance_transform_edt(~A, sampling=voxel_size)
    fwd = dt_to_B[A]
    bwd = dt_to_A[B]
    return float(max(fwd.max(initial=0.0), bwd.max(initial=0.0)))


# ----------------- main perturbation -----------------

def perturb_gt_instances_to_mean_norm_hd(
    gt_labels: np.ndarray,
    target_mean_norm: float,
    voxel_size=(1.0, 1.0, 1.0),
    mode: str = "out",            # "out" bump (FP), "in" dent (FN), "random"
    band_vox: int = 2,            # thickness of patch selection around seed (in voxels)
    avoid_instance_overlap: bool = True,  # when bumping out, only add into background
    report: bool = True,          # compute achieved mean via hausdorff_voxels_mm (slower)
    rng: np.random.Generator | None = None,
):
    """
    Perturb GT instance labels so that the mean over instances of normalize_distance(HD_i) is ~ target_mean_norm.

    - Equal weight per instance (mean of normalized per-instance HD).
    - Creates spatially realistic local bumps/dents using EDT.
    - Returns (perturbed_labels, info_dict) if report else perturbed_labels.

    Known divergence sources:
      - per-instance feasibility caps (r_max)
      - voxel discretization
      - overlap avoidance (bumps restricted to background)
    """
    if rng is None:
        rng = np.random.default_rng()

    gt_labels = np.asarray(gt_labels)
    if gt_labels.ndim != 3:
        raise ValueError("gt_labels must be a 3D volume.")
    voxel_size = np.asarray(voxel_size, dtype=float)
    if voxel_size.size != 3:
        voxel_size = voxel_size[-3:]

    T = float(target_mean_norm)
    if not (0.0 < T <= 1.0):
        raise ValueError("target_mean_norm must be in (0, 1].")

    ids = [int(i) for i in np.unique(gt_labels) if i != 0]
    if not ids:
        out = gt_labels.copy()
        info = {
            "target_mean_norm": T,
            "achieved_mean_norm": 1.0,
            "mean_divergence": 1.0 - T,
            "note": "No instances in GT; nothing perturbed.",
        }
        return (out, info) if report else out

    # ---- choose per-instance feasible minima (in normalized space) ----
    # If mode=="random", use a conservative cap that works for either direction.
    rmax = {}
    smin = {}
    for i in ids:
        M = (gt_labels == i)
        if mode == "in":
            r = _max_radius_in_mm(M, voxel_size)
        elif mode == "out":
            r = _max_radius_out_mm(M, voxel_size)
        else:
            r = min(_max_radius_in_mm(M, voxel_size), _max_radius_out_mm(M, voxel_size))
        rmax[i] = float(r)
        smin[i] = normalize_distance(rmax[i], voxel_size)  # smallest achievable score (worst)

    best_possible_mean = float(np.mean([smin[i] for i in ids]))
    # If target is below what is feasible, best effort = drive all to smin
    if T < best_possible_mean:
        target_s = {i: smin[i] for i in ids}
    else:
        # Allocate drops from 1.0 down toward T, using instances with most drop capacity first
        total_drop = len(ids) * (1.0 - T)
        target_s = {i: 1.0 for i in ids}
        order = sorted(ids, key=lambda i: (1.0 - smin[i]), reverse=True)
        for i in order:
            cap = 1.0 - smin[i]
            use = min(cap, total_drop)
            target_s[i] = 1.0 - use
            total_drop -= use
            if total_drop <= 1e-12:
                break

    # ---- apply perturbations ----
    out = gt_labels.copy()
    background = (out == 0)

    se = _ball_se(max(1, int(band_vox)))
    st = ndi.generate_binary_structure(3, 1)

    per_instance_r = {}
    for i in ids:
        M = (out == i)  # note: uses current out; overlap avoidance keeps this stable
        if not M.any():
            per_instance_r[i] = 0.0
            continue

        # pick direction if random
        dir_mode = mode
        if dir_mode == "random":
            dir_mode = "out" if rng.random() < 0.5 else "in"

        # target radius in mm, capped by feasibility
        r = inv_normalize_distance(target_s[i], voxel_size)
        if r == np.inf:
            # "score 0" request -> use max feasible for that instance
            r = rmax[i]
        else:
            r = min(float(r), rmax[i])
        per_instance_r[i] = float(r)
        if r <= 0.0:
            continue

        S = M & ~ndi.binary_erosion(M, structure=st, iterations=1, border_value=0)
        if not S.any():
            continue

        # seed selection
        if dir_mode == "out":
            seed_flat = _pick_seed_on_surface_max_margin(S, voxel_size, rng)
        else:
            sidx = np.flatnonzero(S)
            seed_flat = int(rng.choice(sidx)) if sidx.size else None
        if seed_flat is None:
            continue

        seeds = np.zeros_like(out, dtype=bool)
        seeds.ravel()[seed_flat] = True

        # patch around seed (band-limited in voxels for locality, then pushed by r in mm)
        # 1) restrict to a local surface patch using voxel-based dilation (cheap + stable)
        patch = S & ndi.binary_dilation(seeds, structure=se, iterations=1)

        # 2) push region by physical radius r using EDT to patch
        dt_patch = ndi.distance_transform_edt(~patch, sampling=voxel_size)

        if dir_mode == "out":
            add = (dt_patch <= r)
            if avoid_instance_overlap:
                add = add & background  # only grow into background
            out[add] = i
            background = (out == 0)
        else:
            rem = (dt_patch <= r) & (out == i)
            out[rem] = 0
            background = (out == 0)

    if not report:
        return out

    # ---- report achieved score + divergence ----
    achieved_scores = []
    requested_scores = []
    achieved_r = {}
    for i in ids:
        A = (gt_labels == i)
        B = (out == i)
        d = hausdorff_voxels_mm(A, B, voxel_size)
        s = normalize_distance(d, voxel_size)
        achieved_scores.append(s)
        requested_scores.append(float(target_s[i]))
        achieved_r[i] = float(d)

    achieved_mean = float(np.mean(achieved_scores)) if achieved_scores else 1.0
    info = {
        "target_mean_norm": T,
        "best_possible_mean_norm": best_possible_mean,
        "achieved_mean_norm": achieved_mean,
        "mean_divergence": achieved_mean - T,
        "per_instance_requested_norm": target_s,
        "per_instance_target_radius_mm": per_instance_r,
        "per_instance_achieved_hd_mm": achieved_r,
        "per_instance_achieved_norm": {i: float(s) for i, s in zip(ids, achieved_scores)},
        "note": (
            "If target < best_possible_mean_norm, target is infeasible; output is best-effort. "
            "Nonzero divergence can also come from discretization and overlap-avoidance."
        ),
    }
    return out, info