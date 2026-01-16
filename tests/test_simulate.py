import numpy as np
from scipy import ndimage as ndi

from cellmap_segmentation_challenge.utils.simulate import (
    perturb_mask_iou_3d,
    perturb_gt_instances_to_mean_norm_hd,
    normalize_distance,
    hausdorff_voxels_mm,
)


def iou(A, B):
    A = np.asarray(A, dtype=bool)
    B = np.asarray(B, dtype=bool)
    inter = np.logical_and(A, B).sum()
    uni = np.logical_or(A, B).sum()
    return float(inter) / float(uni) if uni else 1.0


def test_perturb_mask_iou_3d_hits_target_reasonably():
    rng = np.random.default_rng(0)

    # Make a 3D "organ-like" GT blob (smooth random field thresholded)
    vol = rng.normal(size=(64, 64, 64))
    vol = ndi.gaussian_filter(vol, sigma=2.0)
    G = vol > np.quantile(vol, 0.85)  # ~15% foreground, non-empty

    for t in [0.9, 0.75, 0.6]:
        P = perturb_mask_iou_3d(G, target_iou=t, band=2, p_fn=0.5, rng=rng)
        got = iou(G, P)

        # Allow a small tolerance due to rounding + finite candidate pixels
        assert abs(got - t) <= 0.02, (t, got)

        # Ensure both FP and FN are present (should be, for these t values)
        fn = np.logical_and(G, ~P).sum()
        fp = np.logical_and(~G, P).sum()
        assert fn > 0
        assert fp > 0


def _make_labeled_spheres(shape=(96, 96, 96), voxel_size=(2.0, 1.0, 1.0)):
    """Three non-overlapping labeled spheres inside a 3D volume."""
    Z, Y, X = shape
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    vs = np.asarray(voxel_size, float)

    # centers chosen to be far apart and away from edges
    centers = [(30, 30, 30), (30, 65, 65), (65, 40, 65)]
    radii_mm = [12.0, 10.0, 8.0]  # physical radii

    labels = np.zeros(shape, dtype=np.int32)
    for i, (c, rmm) in enumerate(zip(centers, radii_mm), start=1):
        dz = (zz - c[0]) * vs[0]
        dy = (yy - c[1]) * vs[1]
        dx = (xx - c[2]) * vs[2]
        sphere = (dz * dz + dy * dy + dx * dx) <= (rmm * rmm)
        labels[sphere] = i

    return labels


def test_feasible_target_mean_norm_hd_close_and_report_consistent():
    gt = _make_labeled_spheres()
    voxel_size = (2.0, 1.0, 1.0)
    rng = np.random.default_rng(0)

    target = 0.80
    out, info = perturb_gt_instances_to_mean_norm_hd(
        gt,
        target_mean_norm=target,
        voxel_size=voxel_size,
        mode="out",
        band_vox=2,
        avoid_instance_overlap=True,
        report=True,
        rng=rng,
    )

    # 1) no new IDs
    assert set(np.unique(out)) <= set(np.unique(gt))

    # 2) achieved mean reported and divergence consistent
    assert "achieved_mean_norm" in info
    assert "mean_divergence" in info
    assert abs(info["mean_divergence"] - (info["achieved_mean_norm"] - target)) < 1e-9

    # 3) should be reasonably close (discretization + overlap avoidance can cause error)
    assert abs(info["achieved_mean_norm"] - target) <= 0.10, info


def test_infeasible_target_saturates_to_best_possible():
    gt = _make_labeled_spheres()
    voxel_size = (2.0, 1.0, 1.0)
    rng = np.random.default_rng(1)

    # Extremely low target likely infeasible given finite volume + overlap avoidance
    target = 0.05
    out, info = perturb_gt_instances_to_mean_norm_hd(
        gt,
        target_mean_norm=target,
        voxel_size=voxel_size,
        mode="out",
        band_vox=2,
        avoid_instance_overlap=True,
        report=True,
        rng=rng,
    )

    # Achieved should not go below best possible mean (up to tiny numerical jitter)
    assert info["achieved_mean_norm"] >= info["best_possible_mean_norm"] - 1e-6

    # And since target is lower than feasible, best effort means achieved ~= best_possible
    assert (
        abs(info["achieved_mean_norm"] - info["best_possible_mean_norm"]) <= 0.10
    ), info


def test_random_mode_runs_and_keeps_id_set():
    gt = _make_labeled_spheres()
    voxel_size = (2.0, 1.0, 1.0)
    rng = np.random.default_rng(2)

    out, info = perturb_gt_instances_to_mean_norm_hd(
        gt,
        target_mean_norm=0.75,
        voxel_size=voxel_size,
        mode="random",
        band_vox=3,
        avoid_instance_overlap=True,
        report=True,
        rng=rng,
    )

    assert set(np.unique(out)) <= set(np.unique(gt))
    assert 0.0 <= info["achieved_mean_norm"] <= 1.0


def _make_labeled_spheres(shape=(96, 96, 96), voxel_size=(2.0, 1.0, 1.0)):
    """Three non-overlapping labeled spheres inside the volume, far from edges."""
    Z, Y, X = shape
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    vs = np.asarray(voxel_size, float)

    centers = [(30, 30, 30), (30, 65, 65), (65, 40, 65)]
    radii_mm = [12.0, 10.0, 8.0]

    labels = np.zeros(shape, dtype=np.int32)
    for i, (c, rmm) in enumerate(zip(centers, radii_mm), start=1):
        dz = (zz - c[0]) * vs[0]
        dy = (yy - c[1]) * vs[1]
        dx = (xx - c[2]) * vs[2]
        sphere = (dz * dz + dy * dy + dx * dx) <= (rmm * rmm)
        labels[sphere] = i
    return labels


def _recompute_achieved_mean_norm(gt, out, voxel_size):
    """Independent recomputation from volumes + IDs (GT instances define the average)."""
    ids = [int(i) for i in np.unique(gt) if i != 0]
    if not ids:
        return 1.0

    scores = []
    for i in ids:
        A = gt == i
        B = out == i
        d = hausdorff_voxels_mm(A, B, voxel_size)
        scores.append(normalize_distance(d, voxel_size))
    return float(np.mean(scores))


def test_reported_achieved_mean_norm_matches_independent_recompute():
    gt = _make_labeled_spheres()
    voxel_size = (2.0, 1.0, 1.0)
    rng = np.random.default_rng(0)

    out, info = perturb_gt_instances_to_mean_norm_hd(
        gt,
        target_mean_norm=0.80,
        voxel_size=voxel_size,
        mode="out",
        band_vox=2,
        avoid_instance_overlap=True,
        report=True,
        rng=rng,
    )

    # independent recompute
    recomputed = _recompute_achieved_mean_norm(gt, out, voxel_size)

    # must match the function-reported value very closely
    assert abs(recomputed - info["achieved_mean_norm"]) <= 1e-6, (
        info["achieved_mean_norm"],
        recomputed,
    )

    # divergence also consistent with recomputation
    assert (
        abs((recomputed - info["target_mean_norm"]) - info["mean_divergence"]) <= 1e-6
    )


def test_reported_per_instance_scores_match_independent_recompute_per_id():
    gt = _make_labeled_spheres()
    voxel_size = (2.0, 1.0, 1.0)
    rng = np.random.default_rng(1)

    out, info = perturb_gt_instances_to_mean_norm_hd(
        gt,
        target_mean_norm=0.75,
        voxel_size=voxel_size,
        mode="random",
        band_vox=3,
        avoid_instance_overlap=True,
        report=True,
        rng=rng,
    )

    ids = [int(i) for i in np.unique(gt) if i != 0]
    achieved_map = info["per_instance_achieved_norm"]

    assert set(achieved_map.keys()) == set(ids)

    for i in ids:
        A = gt == i
        B = out == i
        d = hausdorff_voxels_mm(A, B, voxel_size)
        s = normalize_distance(d, voxel_size)
        assert abs(s - achieved_map[i]) <= 1e-6, (i, achieved_map[i], s)


def test_feasible_target_mean_norm_is_reasonably_close():
    gt = _make_labeled_spheres()
    voxel_size = (2.0, 1.0, 1.0)
    rng = np.random.default_rng(2)

    target = 0.80
    out, info = perturb_gt_instances_to_mean_norm_hd(
        gt,
        target_mean_norm=target,
        voxel_size=voxel_size,
        mode="out",
        band_vox=2,
        avoid_instance_overlap=True,
        report=True,
        rng=rng,
    )

    # Use recomputed score (not the reported one) for the "close to target" assertion
    recomputed = _recompute_achieved_mean_norm(gt, out, voxel_size)

    # no new IDs
    assert set(np.unique(out)) <= set(np.unique(gt))

    # should be reasonably close (discretization + overlap-avoidance can cause residual error)
    assert abs(recomputed - target) <= 0.10, (target, recomputed, info)


def test_infeasible_target_saturates_to_best_possible():
    gt = _make_labeled_spheres()
    voxel_size = (2.0, 1.0, 1.0)
    rng = np.random.default_rng(3)

    target = 0.05  # likely infeasible
    out, info = perturb_gt_instances_to_mean_norm_hd(
        gt,
        target_mean_norm=target,
        voxel_size=voxel_size,
        mode="out",
        band_vox=2,
        avoid_instance_overlap=True,
        report=True,
        rng=rng,
    )

    recomputed = _recompute_achieved_mean_norm(gt, out, voxel_size)

    # Achieved cannot be lower than best possible mean (tiny numerical slack)
    assert recomputed >= info["best_possible_mean_norm"] - 1e-6

    # If target is below feasible, output should be near the best possible
    assert abs(recomputed - info["best_possible_mean_norm"]) <= 0.10, (
        recomputed,
        info["best_possible_mean_norm"],
        info,
    )
