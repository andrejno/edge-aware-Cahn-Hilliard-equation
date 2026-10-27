import numpy as np
from scipy import ndimage as ndi

def _stroke(shape, points, width):
    yy, xx = np.indices(shape, dtype=float)
    distance = np.full(shape, np.inf)
    for a, b in zip(points[:-1], points[1:]):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        dy, dx = b - a
        denom = dy * dy + dx * dx
        q = np.clip(((yy - a[0]) * dy + (xx - a[1]) * dx) / max(denom, 1e-15), 0, 1)
        distance = np.minimum(distance, (yy - a[0] - q * dy) ** 2 + (xx - a[1] - q * dx) ** 2)
    return distance <= (width / 2) ** 2

def _paths_image(n, paths, widths):
    scale = n / 96.0
    truth = np.zeros((n, n), bool)
    for pts, width in zip(paths, widths):
        truth |= _stroke(truth.shape, np.asarray(pts) * scale, width * scale)
    return truth

def _rectmask(n, cy, cx, height, width):
    y, x = np.indices((n, n))
    s = n / 96.0
    return (np.abs(y - cy * s) <= height * s / 2) & (np.abs(x - cx * s) <= width * s / 2)

def _point(n, y, x):
    return [int(np.clip(round(y * n / 96), 0, n - 1)), int(np.clip(round(x * n / 96), 0, n - 1))]

def _make_case(split, index, family, n):
    dev = split == 'dev'
    shift = (-2 if dev else 1) + (index % 3 - 1) * 1.2
    metadata = {}
    if family == 'connections':
        gap = ([7, 11] if dev else [6, 8, 10, 12, 14, 16])[index]
        width = ([5, 7] if dev else [4, 5, 6, 7, 5, 6])[index]
        slope = ([0.12, -0.16] if dev else [0, 0.18, -0.18, 0.3, -0.25, 0.08])[index]
        separation = ([17, 20] if dev else [14, 16, 18, 22, 15, 19])[index]
        x = np.linspace(8, 88, 65)
        upper = 48 - separation / 2 + shift + slope * (x - 48)
        lower = 48 + separation / 2 + shift + slope * (x - 48)
        paths = [np.stack([upper, x], axis=1), np.stack([lower, x], axis=1)]
        truth = _paths_image(n, paths, [width, width])
        mask = _rectmask(n, 48, 48, 68, gap)
        left, right = (48 - gap / 2 - 3, 48 + gap / 2 + 3)
        p1 = _point(n, 48 - separation / 2 + shift + slope * (left - 48), left)
        p2 = _point(n, 48 - separation / 2 + shift + slope * (right - 48), right)
        p3 = _point(n, 48 + separation / 2 + shift + slope * (left - 48), left)
        p4 = _point(n, 48 + separation / 2 + shift + slope * (right - 48), right)
        metadata.update(gap=gap, width=width, separation=separation, slope=slope, connectivity_pairs=[{'start': p1, 'end': p2, 'expected': True}, {'start': p3, 'end': p4, 'expected': True}, {'start': p1, 'end': p4, 'expected': False}])
    elif family == 'junctions':
        kind = (['Y', 'T'] if dev else ['Y', 'T', 'L', 'X', 'acute', 'ring'])[index]
        width = ([6, 5] if dev else [5, 7, 5, 6, 5, 6])[index]
        gap = ([11, 13] if dev else [10, 12, 14, 10, 12, 14])[index]
        if kind == 'Y':
            paths = [[(84, 48), (48, 48), (15, 19)], [(48, 48), (15, 77)]]
        elif kind == 'T':
            paths = [[(83, 48), (31, 48)], [(31, 13), (31, 83)]]
        elif kind == 'L':
            paths = [[(16, 36), (64, 36), (64, 82)]]
        elif kind == 'X':
            paths = [[(15, 15), (81, 81)], [(15, 81), (81, 15)]]
        elif kind == 'acute':
            paths = [[(78, 16), (30, 48), (78, 80)]]
        else:
            angle = np.linspace(0, 2 * np.pi, 121)
            paths = [np.stack([48 + 26 * np.sin(angle), 48 + 26 * np.cos(angle)], axis=1)]
        paths = [np.asarray(p, dtype=float) + np.array([shift, -shift]) for p in paths]
        truth = _paths_image(n, paths, [width] * len(paths))
        cy, cx = {'Y': (48, 48), 'T': (31, 48), 'L': (64, 36), 'X': (48, 48), 'acute': (30, 48), 'ring': (48, 74)}[kind]
        mask = _rectmask(n, cy + shift, cx - shift, gap, gap)
        metadata.update(kind=kind, width=width, gap=gap)
    elif family == 'multiscale':
        gap = ([6, 9] if dev else [5, 7, 9, 11, 8, 10])[index]
        bend = ([5, -4] if dev else [0, 3, -3, 6, -6, 4])[index]
        x = np.linspace(7, 89, 81)
        paths = []
        widths = [3, 5, 9]
        for y0, phase in zip([23, 47, 72], [0, 0.4, 0.8]):
            y = y0 + shift + bend * np.sin((x - 7) / 82 * np.pi + phase)
            paths.append(np.stack([y, x], axis=1))
        truth = _paths_image(n, paths, widths)
        mask = _rectmask(n, 48, 41 + index % 2 * 8, 84, gap)
        mask |= _rectmask(n, 49, 70 - index % 3 * 3, 75, 4)
        metadata.update(gap=gap, bend=bend, widths=widths)
    elif family == 'noise':
        gap = ([6, 8] if dev else [5, 7, 9, 6, 8, 10])[index]
        paths = [[(80, 47), (45, 48), (14, 20)], [(45, 48), (16, 78)], [(64, 47), (71, 76)], [(28, 33), (25, 12)]]
        paths = [np.asarray(p, dtype=float) + np.array([shift, -shift]) for p in paths]
        widths = [6, 5, 4, 3]
        truth = _paths_image(n, paths, widths)
        mask = _rectmask(n, 45 + shift, 48 - shift, gap + 2, gap)
        mask |= _rectmask(n, 66, 66, 8, 8)
        metadata.update(gap=gap, widths=widths, noise_rate=([0.01, 0.03] if dev else [0.01, 0.01, 0.02, 0.02, 0.03, 0.03])[index])
    else:
        raise ValueError(family)
    phase = 2 * truth.astype(float) - 1
    observed = phase.copy()
    seed = 202610260 + (0 if dev else 1000) + 100 * ['connections', 'junctions', 'multiscale', 'noise'].index(family) + index
    if family == 'noise':
        rng = np.random.default_rng(seed)
        flips = (rng.random(truth.shape) < metadata['noise_rate']) & ~mask
        observed[flips] *= -1
        metadata['flipped_pixel_count'] = int(flips.sum())
    observed[mask] = 0
    metadata.update(seed=seed, size=n, missing_pixels=int(mask.sum()), foreground_missing_pixels=int((truth & mask).sum()))
    return dict(id=f'{split}_{family}_{index:02d}', split=split, family=family, truth=truth, mask=mask, observed=observed, initial=observed.copy(), metadata=metadata)

def generate_cases(split='dev', size=96):
    if split not in ('dev', 'test'):
        raise ValueError("split must be 'dev' or 'test'")
    count = 2 if split == 'dev' else 6
    return [_make_case(split, k, family, int(size)) for family in ('connections', 'junctions', 'multiscale', 'noise') for k in range(count)]

def skeletonize(binary):
    a = np.pad(np.asarray(binary, dtype=bool), 1)
    for iteration in range(max(a.shape) * 2):
        changed = False
        for step in (0, 1):
            p2 = a[:-2, 1:-1]
            p3 = a[:-2, 2:]
            p4 = a[1:-1, 2:]
            p5 = a[2:, 2:]
            p6 = a[2:, 1:-1]
            p7 = a[2:, :-2]
            p8 = a[1:-1, :-2]
            p9 = a[:-2, :-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum((p.astype(np.uint8) for p in neighbors))
            transitions = sum(((~neighbors[j] & neighbors[(j + 1) % 8]).astype(np.uint8) for j in range(8)))
            if step == 0:
                groups = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                groups = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            remove = a[1:-1, 1:-1] & (count >= 2) & (count <= 6) & (transitions == 1) & groups
            if remove.any():
                a[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            break
    return a[1:-1, 1:-1].copy()

def _fscore(precision, recall):
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0

def _cldice(pred, truth, region=None):
    sp = skeletonize(pred)
    st = skeletonize(truth)
    if region is not None:
        sp &= region
        st &= region
    if not sp.any() and (not st.any()):
        return 1.0
    if not sp.any() or not st.any():
        return 0.0
    return _fscore((sp & truth).sum() / sp.sum(), (st & pred).sum() / st.sum())

def metrics(u, truth, mask, boundary_tolerance=1.5):
    field = np.asarray(u, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    pred = field > 0
    truth_phase = 2 * truth.astype(float) - 1
    tp = int(np.sum(pred & truth & mask))
    fp = int(np.sum(pred & ~truth & mask))
    fn = int(np.sum(~pred & truth & mask))
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    bp = pred & ~ndi.binary_erosion(pred, structure=np.ones((3, 3)), border_value=0)
    bt = truth & ~ndi.binary_erosion(truth, structure=np.ones((3, 3)), border_value=0)
    if bp.any() and bt.any():
        d_to_t = ndi.distance_transform_edt(~bt)
        d_to_p = ndi.distance_transform_edt(~bp)
        bpr = bp & mask
        btr = bt & mask
        p = float(np.mean(d_to_t[bpr] <= boundary_tolerance)) if bpr.any() else 0.0
        r = float(np.mean(d_to_p[btr] <= boundary_tolerance)) if btr.any() else 0.0
        boundary_f1 = _fscore(p, r)
    else:
        boundary_f1 = 1.0 if not bp.any() and (not bt.any()) else 0.0
    return dict(masked_dice=float(dice), masked_iou=float(iou), masked_rmse=float(np.sqrt(np.mean((field[mask] - truth_phase[mask]) ** 2))) if mask.any() else 0.0, outside_rmse=float(np.sqrt(np.mean((field[~mask] - truth_phase[~mask]) ** 2))) if (~mask).any() else 0.0, cldice=float(_cldice(pred, truth)), masked_cldice=float(_cldice(pred, truth, mask)), boundary_f1=float(boundary_f1), phase_overshoot=float(np.maximum(np.abs(field) - 1, 0).max()), masked_tp=tp, masked_fp=fp, masked_fn=fn)

def evaluate(u, case):
    out = metrics(u, case['truth'], case['mask'])
    obs = case.get('observed', 2 * case['truth'].astype(float) - 1)
    known = ~case['mask']
    out['observed_fidelity_rmse'] = float(np.sqrt(np.mean((u[known] - obs[known]) ** 2))) if known.any() else 0.0
    pairs = case.get('metadata', {}).get('connectivity_pairs', [])
    if pairs:
        labels, n = ndi.label(np.asarray(u) > 0, structure=np.ones((3, 3)))
        true_connections = []
        false_bridges = []
        for pair in pairs:
            a = tuple(pair['start'])
            b = tuple(pair['end'])
            connected = bool(labels[a] > 0 and labels[a] == labels[b])
            if pair['expected']:
                true_connections.append(connected)
            else:
                false_bridges.append(connected)
        out['connection_recovery'] = float(np.mean(true_connections)) if true_connections else float('nan')
        out['false_bridge_rate'] = float(np.mean(false_bridges)) if false_bridges else float('nan')
    return out
if __name__ == '__main__':
    from pathlib import Path
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(__file__).parent / 'benchmark_design'
    out.mkdir(exist_ok=True)
    cases = generate_cases('test')
    fig, axes = plt.subplots(4, 6, figsize=(12, 8))
    for ax, case in zip(axes.ravel(), cases):
        image = (case['observed'] + 1) / 2
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(case['id'].replace('test_', ''), fontsize=8)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / 'test_scene_gallery.png', dpi=180)
    fig.savefig(out / 'test_scene_gallery.pdf')
    plt.close(fig)
    summary = [{k: v for k, v in case.items() if k not in ('truth', 'mask', 'observed', 'initial')} for case in generate_cases('dev') + cases]
    (out / 'case_manifest.json').write_text(json.dumps(summary, indent=2))
    print('Wrote fixed manifest and gallery for', len(cases), 'test scenes')
