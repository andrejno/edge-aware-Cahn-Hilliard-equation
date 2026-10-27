from pathlib import Path
import json
import re
import sys
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import results as r

def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value

def image(array, name, mask=None, continuous=False):
    ink = np.array([28, 51, 59], dtype=np.uint8)
    white = np.array([250, 250, 247], dtype=np.uint8)
    if continuous:
        weight = np.clip((array[..., None] + 1) / 2, 0, 1)
        pixels = np.round(weight * ink + (1 - weight) * white).astype(np.uint8)
    else:
        pixels = np.where(array[..., None] > 0, ink, white)
    if mask is not None:
        pixels[mask] = [210, 219, 219]
    Image.fromarray(pixels).save(ROOT / 'assets' / 'images' / name)

def outline(mask, name):
    h, w = mask.shape
    segments = []
    for y, x in zip(*np.where(mask)):
        if y == 0 or not mask[y - 1, x]: segments.append(f'M{x},{y}h1')
        if y == h - 1 or not mask[y + 1, x]: segments.append(f'M{x},{y+1}h1')
        if x == 0 or not mask[y, x - 1]: segments.append(f'M{x},{y}v1')
        if x == w - 1 or not mask[y, x + 1]: segments.append(f'M{x+1},{y}v1')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path d="{"".join(segments)}" fill="none" stroke="#35a296" stroke-width="0.25"/></svg>'
    (ROOT / 'assets' / 'images' / name).write_text(svg)

def main():
    r.style()
    cases = json.loads((ROOT / 'data' / 'cases.json').read_text())
    rows = {d: r.records(d) for d in ['synthetic', 'retina']}
    case_list = []
    for entry in cases:
        case_id, dataset = entry['id'], entry['dataset']
        case = r.case_data(case_id)
        image(case['observed'], f'{case_id}_observed.png', case['mask'], continuous=True)
        image(case['truth'], f'{case_id}_reference.png')
        outline(case['mask'], f'{case_id}_mask.svg')
        if 'rgb' in case:
            Image.fromarray(np.round(255 * case['rgb']).astype(np.uint8)).save(ROOT / 'assets' / 'images' / f'{case_id}_rgb.jpg', quality=92)
        scores = {}
        for row in [x for x in rows[dataset] if x['case_id'] == case_id]:
            method = row['method']
            scores[method] = row['scores']
            u = r.field(dataset, case_id, method)
            image(u, f'{case_id}_{method}.png')
            rgba = np.zeros((*u.shape, 4), dtype=np.uint8)
            rgba[(u > 0) & ~case['truth'] & case['mask']] = [199, 92, 67, 230]
            rgba[(u <= 0) & case['truth'] & case['mask']] = [49, 117, 180, 230]
            Image.fromarray(rgba).save(ROOT / 'assets' / 'images' / f'{case_id}_{method}_error.png')
        case_list.append(dict(**entry, scores=scores, vk_scores={x['method']: x.get('vk_scores') for x in rows[dataset] if x['case_id'] == case_id}))
    summary = {}
    widths = {}
    for dataset in ['synthetic', 'retina']:
        summary[dataset] = r.means(dataset).reset_index().to_dict('records')
    summary['retina_VK'] = r.means('retina', 'VK').reset_index().to_dict('records')
    for reference in ['AH', 'VK']:
        widths[reference] = {str(w): r.means('retina', reference, w).reset_index().to_dict('records') for w in [4, 8]}
    contrasts = {}
    for name in ['summary_synthetic', 'summary_retina_AH', 'summary_retina_VK', 'summary_retina_AH_w4', 'summary_retina_AH_w8']:
        contrasts[name] = [x for x in r.read(name + '.json')['comparisons'] if x['metric'] == 'masked_dice']
    calibration = {}
    for dataset in ['synthetic', 'retina']:
        p = np.load(ROOT / 'data' / 'records' / f'{dataset}_width_calibration_profiles.npz')
        calibration[dataset] = dict(r.read(f'{dataset}_width_calibration.json'), profiles={k: p[k].tolist() for k in p.files})
    refinements = []
    for dataset in ['synthetic', 'retina']:
        refinements.extend(r.read(f'{dataset}_summary_refinement.json')['comparisons'])
    data = {'methods': r.METHODS, 'colors': r.COLORS, 'cases': case_list, 'summary': summary, 'families': r.family_means().reset_index().to_dict('records'), 'widths': widths, 'contrasts': contrasts, 'diagnostics': r.diagnostics().to_dict('records'), 'tv': r.tv_accuracy().to_dict('records'), 'parameters': r.selected_parameters().to_dict('records'), 'tuning': r.tuning_table().to_dict('records'), 'calibration': calibration, 'refinement': refinements, 'extended': {name: r.read(name + '_refinement.json') for name in ['sensitive_time_summary', 'retina_sensitive_time_summary', 'fine_grid_time_summary']}, 'spatial': pd.read_csv(ROOT / 'data' / 'records' / 'manufactured_convergence.csv').to_dict('records'), 'temporal': r.read('solver_validation.json')['convergence'], 'verification': r.read('verification_summary.json')}
    data = clean(data)
    (ROOT / 'assets' / 'data' / 'results.js').write_text('window.CHPM_DATA = ' + json.dumps(data, separators=(',', ':'), allow_nan=False) + ';\n')
    pd.concat([r.frame('synthetic'), r.frame('retina')], ignore_index=True).to_csv(ROOT / 'assets' / 'data' / 'all_case_metrics.csv', index=False)
    r.frame('retina', 'VK').to_csv(ROOT / 'assets' / 'data' / 'retinal_VK_metrics.csv', index=False)
    r.tuning_table().to_csv(ROOT / 'assets' / 'data' / 'development_selection.csv', index=False)
    for name, function in [('scores', r.score_figure), ('families', r.family_figure), ('retina', r.retinal_figure), ('coefficient_controls', r.calibration_figure)]:
        fig = function()
        fig.savefig(ROOT / 'assets' / 'figures' / f'{name}.svg', bbox_inches='tight')
        fig.savefig(ROOT / 'assets' / 'figures' / f'{name}.png', dpi=600, bbox_inches='tight')
        path = ROOT / 'assets' / 'figures' / f'{name}.svg'
        path.write_text(re.sub(r'<!--.*?-->', '', path.read_text(), flags=re.S))
        r.plt.close(fig)
    print(json.dumps({'cases': len(cases), 'reconstructions': sum(len(c['scores']) for c in case_list)}))

if __name__ == '__main__':
    main()
