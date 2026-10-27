from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .prepare_retina import load_cases
except ImportError:
    from prepare_retina import load_cases

def crossed_scratches(shape=(192, 192), width=4):
    if tuple(shape) != (192, 192):
        raise ValueError('The frozen retinal protocol uses native 192 x 192 crops')
    if width not in (4, 8):
        raise ValueError('The frozen scratch widths are 4 and 8 pixels')
    h, w = shape
    y, x = np.mgrid[:h, :w]
    slope = 0.85
    denominator = np.sqrt(1.0 + slope * slope)
    distance_0 = np.abs(y - slope * (x - 0.5 * (w - 1)) - 0.48 * (h - 1)) / denominator
    distance_1 = np.abs(y + slope * (x - 0.5 * (w - 1)) - 0.52 * (h - 1)) / denominator
    mask = (distance_0 < width / 2) | (distance_1 < width / 2)
    mask[:3, :] = False
    mask[-3:, :] = False
    mask[:, :3] = False
    mask[:, -3:] = False
    return mask

def generate_retina_cases(split='dev', data_dir=None):
    split_mapping = {'dev': 'development', 'development': 'development', 'test': 'test'}
    if split not in split_mapping:
        raise ValueError("split must be 'dev' or 'test'")
    source_split = split_mapping[split]
    short_split = 'dev' if source_split == 'development' else 'test'
    cases = []
    for crop in load_cases(data_dir=data_dir, split=source_split):
        truth = np.asarray(crop['ah'], dtype=bool)
        vk = np.asarray(crop['vk'], dtype=bool)
        for width in (4, 8):
            mask = crossed_scratches(truth.shape, width)
            observed = 2.0 * truth.astype(np.float64) - 1.0
            observed[mask] = 0.0
            cases.append({'id': f"{short_split}_retina_{crop['case_id']}_w{width}", 'family': f'retina_w{width}', 'split': short_split, 'truth': truth.copy(), 'mask': mask, 'observed': observed, 'initial': observed.copy(), 'metadata': {'image_id': crop['image_id'], 'cropid': crop['case_id'], 'crop_id': crop['case_id'], 'width': width, 'split': short_split, 'bbox_xyxy': list(crop['bbox_xyxy']), 'mask_fraction': float(mask.mean()), 'mask_pixel_count': int(mask.sum()), 'scratch_slope_magnitude': 0.85, 'scratch_y_fractions': [0.48, 0.52], 'observed_boundary_width': 3, 'primary_annotation': 'AH', 'secondary_annotation': 'VK', 'independent_sampling_unit': crop['image_id']}, 'rgb': crop['rgb'].copy(), 'vk': vk.copy()})
    return cases

def write_manifest(path=None, data_dir=None):
    if path is None:
        path = Path(__file__).resolve().parent / 'benchmark_design' / 'retina_case_manifest.json'
    path = Path(path)
    records = []
    for case in generate_retina_cases('dev', data_dir) + generate_retina_cases('test', data_dir):
        record = {key: case[key] for key in ('id', 'family', 'split', 'metadata')}
        for field in ('mask', 'truth', 'vk'):
            pixels = np.ascontiguousarray(case[field], dtype=np.uint8).tobytes()
            record[f'{field}_uint8_sha256'] = hashlib.sha256(pixels).hexdigest()
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'protocol': 'STARE native192 fixed crossed-scratch label inpainting', 'source_manifest': 'experiments/data/retina/data_manifest.json', 'mask_convention': 'True denotes occluded pixels, observed value and initial value are zero there', 'width_4_mask_pixels': 1936, 'width_8_mask_pixels': 3840, 'case_count': len(records), 'case_counts_by_split': {'dev': sum((r['split'] == 'dev' for r in records)), 'test': sum((r['split'] == 'test' for r in records))}, 'statistical_unit': 'source image, with four cases per image averaged before statistical inference', 'cases': records}, indent=2) + '\n')
    return path
if __name__ == '__main__':
    manifest_path = write_manifest()
    print(f'Saved 80 retinal case records to {manifest_path}')
