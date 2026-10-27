from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen
import numpy as np
from PIL import Image
BASE_URL = 'http://cecas.clemson.edu/~ahoover/stare/probing/'
ARCHIVES = {'stare-images.tar': ('images', '5f7b509b6067cad4f1be84933145d783de9ec087b3eaaf4db0103dd0144dd433'), 'labels-ah.tar': ('labels_ah', 'ebf2f1e17ca955f24579d9edd990e2dae79a5c82def69f0985d8e24f826ddd2f'), 'labels-vk.tar': ('labels_vk', '47474a701536b0cfdb369fdce012be36141e9f44d80387f0179446b5cb0f5576')}
DEVELOPMENT_IDS = {'im0001', 'im0004', 'im0077', 'im0162', 'im0236', 'im0255'}

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def prepare(output: Path, crop_size: int=192) -> dict:
    if crop_size <= 0:
        raise ValueError('crop_size must be positive')
    raw = output / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': 'STARE vessel segmentation subset', 'source_url': BASE_URL + 'index.html', 'source_overview': 'http://cecas.clemson.edu/~ahoover/stare/', 'retrieved_date': '2026-10-26', 'citation': {'authors': 'A. Hoover, V. Kouznetsova, M. Goldbaum', 'title': 'Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response', 'journal': 'IEEE Transactions on Medical Imaging', 'volume': 19, 'issue': 3, 'pages': '203-210', 'year': 2000, 'doi': '10.1109/42.845178'}, 'terms': "The official authors' site makes these image and annotation archives publicly downloadable for the published research benchmark. The download pages do not specify a Creative Commons or other explicit redistribution license. Preserve attribution and obtain clarification from the dataset owners for redistribution or other uses as needed. Release this downloader and hashes, rather than the source pixels.", 'annotation_roles': {'ah': 'Adam Hoover hand labeling, primary reconstruction reference', 'vk': 'Valentina Kouznetsova hand labeling, independent annotation sensitivity reference'}, 'preprocessing': 'Lossless PPM decoding to uint8 PNG. RGB photographs retain all three channels. The two original binary labels have values 0 and 255. No interpolation, normalization, resizing or image selection.', 'crop_protocol': {'size': crop_size, 'number_per_image': 2, 'centers_xy': '(floor(width/3), floor(height/2)) and (floor(2*width/3), floor(height/2))', 'coordinate_convention': 'zero-based half-open [y0:y1,x0:x1]', 'selection': 'Fixed spatial coordinates independent of labels and outcomes'}, 'split_protocol': {'development': sorted(DEVELOPMENT_IDS), 'test': [], 'unit': 'source image', 'selection': 'Six deterministic image IDs, spread through the sorted 20-image list, were designated for development before solver evaluation. The remaining fourteen are held out. Both crops and both annotations of an image share split membership. Clinical characteristics were not used for stratification.'}, 'archives': [], 'images': [], 'cases': []}
    decoded: dict[str, dict[str, np.ndarray]] = {}
    for archive_name, (kind, expected_hash) in ARCHIVES.items():
        archive_path = raw / archive_name
        if not archive_path.exists():
            archive_path.write_bytes(urlopen(BASE_URL + archive_name, timeout=90).read())
        blob = archive_path.read_bytes()
        digest = sha256(blob)
        if digest != expected_hash:
            raise ValueError(f'Archive checksum mismatch: {archive_name}')
        manifest['archives'].append({'filename': archive_name, 'url': BASE_URL + archive_name, 'sha256': digest, 'size_bytes': len(blob)})
        (output / kind).mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:') as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith('.ppm.gz'):
                    continue
                image_id = Path(member.name).name.split('.')[0]
                if not image_id.startswith('im') or not image_id[2:].isdigit():
                    raise ValueError(f'Unexpected image ID {image_id}')
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f'Missing archive content {member.name}')
                ppm = gzip.decompress(source.read())
                with Image.open(io.BytesIO(ppm)) as image:
                    array = np.array(image)
                    if kind != 'images':
                        if array.ndim != 2 or set(np.unique(array)) != {0, 255}:
                            raise ValueError(f'Nonbinary label {member.name}')
                    elif array.ndim != 3 or array.shape[2] != 3:
                        raise ValueError(f'Unexpected photograph shape {array.shape}')
                    image.save(output / kind / f'{image_id}.png')
                decoded.setdefault(image_id, {})[kind] = array
    if len(decoded) != 20:
        raise ValueError(f'Expected 20 source images, found {len(decoded)}')
    for image_id, arrays in sorted(decoded.items()):
        if set(arrays) != {'images', 'labels_ah', 'labels_vk'}:
            raise ValueError(f'Incomplete source {image_id}')
        h, w = arrays['images'].shape[:2]
        if arrays['labels_ah'].shape != (h, w) or arrays['labels_vk'].shape != (h, w):
            raise ValueError(f'Image and label shapes disagree for {image_id}')
        split = 'development' if image_id in DEVELOPMENT_IDS else 'test'
        if split == 'test':
            manifest['split_protocol']['test'].append(image_id)
        image_record = {'image_id': image_id, 'split': split, 'height': h, 'width': w, 'files': {}}
        for kind, array in arrays.items():
            image_record['files'][kind] = {'path': f'{kind}/{image_id}.png', 'png_sha256': sha256((output / kind / f'{image_id}.png').read_bytes()), 'pixels_sha256': sha256(np.ascontiguousarray(array).tobytes())}
        manifest['images'].append(image_record)
        for crop_index, cx in enumerate((w // 3, 2 * w // 3)):
            x0 = cx - crop_size // 2
            y0 = h // 2 - crop_size // 2
            x1, y1 = (x0 + crop_size, y0 + crop_size)
            if x0 < 0 or y0 < 0 or x1 > w or (y1 > h):
                raise ValueError('Crop does not fit source image')
            case_id = f'{image_id}_c{crop_index}'
            target = output / 'crops' / case_id
            target.mkdir(parents=True, exist_ok=True)
            record = {'case_id': case_id, 'image_id': image_id, 'split': split, 'bbox_xyxy': [x0, y0, x1, y1], 'files': {}}
            for kind, array in arrays.items():
                crop = array[y0:y1, x0:x1].copy()
                path = target / f'{kind}.png'
                Image.fromarray(crop).save(path)
                record['files'][kind] = {'path': str(path.relative_to(output)), 'png_sha256': sha256(path.read_bytes()), 'pixels_sha256': sha256(crop.tobytes())}
            manifest['cases'].append(record)
    (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest

def load_cases(data_dir: Path | str | None=None, split: str | None=None):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / 'data' / 'retina'
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / 'data_manifest.json').read_text())
    for record in manifest['cases']:
        if split is not None and record['split'] != split:
            continue
        case = {key: record[key] for key in ('case_id', 'image_id', 'split', 'bbox_xyxy')}
        for kind, field in (('images', 'rgb'), ('labels_ah', 'ah'), ('labels_vk', 'vk')):
            with Image.open(data_dir / record['files'][kind]['path']) as image:
                case[field] = np.asarray(image, dtype=np.float64) / 255.0
        yield case
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'data' / 'retina')
    parser.add_argument('--crop-size', type=int, default=192)
    args = parser.parse_args()
    result = prepare(args.output, args.crop_size)
    print(f"Prepared {len(result['images'])} images and {len(result['cases'])} native-resolution crops")
    print(f"Development IDs: {', '.join(result['split_protocol']['development'])}")
    print(f"Test IDs: {', '.join(result['split_protocol']['test'])}")
