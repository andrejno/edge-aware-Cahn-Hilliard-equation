from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from benchmarks import generate_cases
from retina_cases import generate_retina_cases
from prepare_retina import prepare

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retina-source', type=Path, default=ROOT / 'data' / 'stare')
    args = parser.parse_args()
    if not (args.retina_source / 'crops').exists():
        prepare(args.retina_source)
    cases = [('synthetic', c) for c in generate_cases('test', 96)]
    cases += [('retina', c) for c in generate_retina_cases('test', args.retina_source)]
    records = []
    for dataset, case in cases:
        fields = {k: case[k] for k in ['truth', 'mask', 'observed', 'initial', 'rgb', 'vk'] if k in case}
        np.savez_compressed(ROOT / 'data' / 'cases' / (case['id'] + '.npz'), **fields)
        fingerprint = hashlib.sha256(np.ascontiguousarray(case['observed'], dtype=np.float64).tobytes() + np.ascontiguousarray(case['mask']).tobytes()).hexdigest()
        records.append({'dataset': dataset, 'id': case['id'], 'family': case['family'], 'metadata': case['metadata'], 'input_sha256': fingerprint})
    (ROOT / 'data' / 'cases.json').write_text(json.dumps(records, indent=2))
    print(json.dumps({'synthetic': 24, 'retina': 56, 'cases': len(records)}))

if __name__ == '__main__':
    main()
