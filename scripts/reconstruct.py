from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chpm_solver import Params, solve
from baselines import biharmonic, tv_inpaint

def reconstruct(row, force=False):
    dataset, method, case_id = row['dataset'], row['method'], row['case_id']
    path = ROOT / 'data' / 'fields' / dataset / method / (case_id + '.npz')
    case = dict(np.load(ROOT / 'data' / 'cases' / (case_id + '.npz')))
    fingerprint = hashlib.sha256(np.ascontiguousarray(case['observed'], dtype=np.float64).tobytes() + np.ascontiguousarray(case['mask']).tobytes()).hexdigest()
    if fingerprint != row['input_sha256']:
        raise ValueError(case_id)
    if path.exists() and not force:
        u = np.load(path)['u'].astype(float)
    else:
        if method == 'Biharmonic':
            u, diagnostics = biharmonic(case['observed'], case['mask'])
        elif method == 'TV':
            u, diagnostics = tv_inpaint(case['observed'], case['mask'], iterations=10000, tolerance=2e-6)
        else:
            u, diagnostics = solve(case['initial'], case['observed'], case['mask'], Params(**row['diagnostics']['parameters']))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, u=u.astype(np.float32))
    m, truth = case['mask'], case['truth']
    pred = u > 0
    dice = 2 * np.sum(pred & truth & m) / (np.sum(pred & m) + np.sum(truth & m))
    rmse = np.sqrt(np.mean((u[m] - (2 * truth[m].astype(float) - 1)) ** 2))
    differences = [abs(dice - row['scores']['masked_dice']), abs(rmse - row['scores']['masked_rmse'])]
    if differences[0] > 1e-12 or differences[1] > 1e-6:
        raise ValueError((case_id, method, differences))
    return {'case_id': case_id, 'method': method, 'maximum_difference': max(differences)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['synthetic', 'retina', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    datasets = ['synthetic', 'retina'] if args.dataset == 'all' else [args.dataset]
    rows = [r for d in datasets for r in json.loads((ROOT / 'data' / 'records' / f'{d}_test_rows.json').read_text())]
    checks = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(reconstruct, row, args.force) for row in rows]
        for future in as_completed(futures):
            checks.append(future.result())
            if len(checks) % 24 == 0 or len(checks) == len(rows):
                print(f'{len(checks)}/{len(rows)}', flush=True)
    print(json.dumps({'reconstructions': len(checks), 'maximum_score_difference': max(c['maximum_difference'] for c in checks)}))

if __name__ == '__main__':
    main()
