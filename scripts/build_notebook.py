from pathlib import Path
import html
import json
import re
import sys
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import mistune

ROOT = Path(__file__).resolve().parents[1]
cells = []

def heading(text):
    cells.append(nbformat.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbformat.v4.new_code_cell(text.strip()))

heading('# CHPM\n\nMartin Ehler · Andrej Novak\n\nNumerical results for edge-aware Cahn–Hilliard inpainting.')
code('''
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib_inline.backend_inline import configure_inline_support

configure_inline_support(get_ipython(), 'inline')
ROOT = next(p for p in [Path.cwd(), Path.cwd().parent] if (p / 'data' / 'records').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
import results as r
import verification as v

r.style()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.6g}')
cases = json.loads((ROOT / 'data' / 'cases.json').read_text())
''')
heading('## Parameters')
code("display(r.selected_parameters())")
heading('## Development selection')
code("tuning = r.tuning_table()\ndisplay(tuning[tuning.Rank <= 3])")
heading('### Complete development rankings')
code("display(tuning)")
heading('## Synthetic geometries')
code("display(r.means('synthetic').rename(columns=r.NAMES))\nfig = r.score_figure()\ndisplay(fig)\nplt.close(fig)")
heading('### Geometry families')
code("display(r.family_means().rename(columns=r.NAMES))\nfig = r.family_figure()\ndisplay(fig)\nplt.close(fig)")
heading('### Paired differences')
code('''
synthetic_inference = r.inference('synthetic')
display(synthetic_inference)
recorded = r.read('summary_synthetic.json')['comparisons']
for row in synthetic_inference.to_dict('records'):
    expected = next(x for x in recorded if x['method_b'] == row['Comparator'] and x['metric'] == 'masked_dice')
    np.testing.assert_allclose([row['Difference'], row['95% lower'], row['95% upper']], [expected['mean_difference'], expected['ci_low'], expected['ci_high']], rtol=0, atol=1e-14)
''')
heading('### Endpoint connections')
code('''
connection_rows = []
for method in r.METHODS:
    group = [x for x in r.records('synthetic') if x['family'] == 'connections' and x['method'] == method]
    connection_rows.append({'Method': method, 'Recovered endpoint pairs': int(round(sum(2 * x['scores']['connection_recovery'] for x in group))), 'Cross-strip bridges': int(round(sum(x['scores']['false_bridge_rate'] for x in group)))})
display(pd.DataFrame(connection_rows))
''')
heading('## Retinal vessel annotations')
heading('### AH reference')
code("display(r.means('retina').rename(columns=r.NAMES))\nretinal_inference = r.inference('retina')\ndisplay(retinal_inference)\nfig = r.retinal_figure('AH')\ndisplay(fig)\nplt.close(fig)")
heading('### VK reference')
code("display(r.means('retina', 'VK').rename(columns=r.NAMES))\ndisplay(r.inference('retina', 'VK'))\nfig = r.retinal_figure('VK')\ndisplay(fig)\nplt.close(fig)")
heading('### Scratch width')
code('''
width_scores = []
for reference in ['AH', 'VK']:
    for width in [4, 8]:
        table = r.means('retina', reference, width).reset_index()
        table.insert(0, 'Reference', reference)
        table.insert(1, 'Width', width)
        width_scores.append(table)
display(pd.concat(width_scores, ignore_index=True).rename(columns=r.NAMES))
''')
heading('### Four-pixel scratches')
code("display(r.inference('retina', 'AH', 4))\ndisplay(r.inference('retina', 'VK', 4))")
heading('### Eight-pixel scratches')
code("display(r.inference('retina', 'AH', 8))\ndisplay(r.inference('retina', 'VK', 8))")
heading('## Coefficient controls')
code('''
controls = []
calibration = []
for dataset in ['synthetic', 'retina']:
    table = r.means(dataset).loc[['CHPM', 'CH-width', 'Frozen']].reset_index()
    table.insert(0, 'Dataset', dataset)
    controls.append(table)
    c = r.read(f'{dataset}_width_calibration.json')
    calibration.append({'Dataset': dataset, 'a0': c['constant_a'], 'CHPM width': c['target_width'], 'CH-width width': c['matched_width'], 'Width difference': c['absolute_width_error']})
display(pd.concat(controls, ignore_index=True).rename(columns=r.NAMES))
display(pd.DataFrame(calibration))
fig = r.calibration_figure()
display(fig)
plt.close(fig)
''')
heading('## Observed-region errors')
code("display(r.diagnostics())")
heading('### Execution times\n\nMedian elapsed seconds from the recorded parallel runs. Cahn–Hilliard timing covers time stepping; biharmonic timing includes matrix assembly and the linear solve; TV timing includes initialization, iterations and the final dual-gap calculation.')
code('''
times = []
for dataset in ['synthetic', 'retina']:
    for method in r.METHODS:
        values = [row['diagnostics']['seconds'] for row in r.records(dataset) if row['method'] == method]
        times.append({'Dataset': dataset, 'Method': method, 'Median elapsed seconds': np.median(values)})
display(pd.DataFrame(times))
''')
heading('### Noisy observations')
code("noise = r.frame('synthetic').query(\"family == 'noise'\")\ndisplay(noise.groupby('method')[['outside_rmse', 'observed_fidelity_rmse', 'masked_dice']].mean().reindex(r.METHODS).rename(columns=r.NAMES))")
heading('## TV accuracy')
code('''
display(r.tv_accuracy())
for dataset in ['synthetic', 'retina']:
    for row in r.records(dataset):
        if row['method'] == 'TV':
            diagnostic = row['diagnostics']
            gap = max(diagnostic['tv'] - diagnostic['dual_lower_bound'], 0) / max(1, diagnostic['tv'])
            np.testing.assert_allclose(gap, diagnostic['relative_primal_dual_gap'], rtol=0, atol=1e-15)
''')
heading('## Temporal and spatial refinement')
heading('### Synthetic time-step refinement')
code("columns = ['case_id', 'method', 'original_dt', 'refined_dt', 'original_dice', 'refined_dice', 'field_rms', 'masked_threshold_disagreement']\ndisplay(r.refinement_table('synthetic')[columns])")
heading('### Synthetic spatial refinement')
code("columns = ['case_id', 'method', 'original_h', 'refined_h', 'original_dice', 'refined_dice', 'field_rms', 'masked_threshold_disagreement']\ndisplay(r.refinement_table('synthetic', 'grid')[columns])")
heading('### Retinal time-step refinement')
code("columns = ['case_id', 'method', 'original_dt', 'refined_dt', 'original_dice', 'refined_dice', 'field_rms', 'masked_threshold_disagreement']\ndisplay(r.refinement_table('retina')[columns])")
heading('### Further time-step refinement')
code('''
extended = []
for name in ['sensitive_time_summary', 'retina_sensitive_time_summary']:
    summary = r.read(name + '_refinement.json')
    for comparison in summary['comparisons']:
        extended.append({'case_id': summary['case_id'], 'method': summary['method'], **comparison})
display(pd.DataFrame(extended))
fine = r.read('fine_grid_time_summary_refinement.json')
display(pd.DataFrame([fine]))
a = np.load(ROOT / 'data' / 'refinement' / 'synthetic__test_multiscale_04__CHPM__h0.5__dt0.05.npz')
b = np.load(ROOT / 'data' / 'refinement' / 'synthetic__test_multiscale_04__CHPM__h0.5__dt0.025.npz')
display(pd.DataFrame([{'Grid': '96 x 96 block averages', 'RMS difference': np.sqrt(np.mean((a['averaged'] - b['averaged']) ** 2))}]))
''')
heading('## Discrete mass balance and convergence')
heading('### Manufactured solution\n\nU(x,t) = 0.1 + 0.2 exp(−t) cos(πx), x ∈ (0,1), T = 0.05.')
code('''
spatial, tolerance_difference = v.manufactured()
display(spatial)
fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
ax.loglog(spatial.h, spatial.L2_error, 'o-', color=r.COLORS['CHPM'], label='Computed error')
ax.loglog(spatial.h, spatial.L2_error.iloc[-1] * (spatial.h / spatial.h.iloc[-1]) ** 2, '--', color=r.COLORS['CH'], label='Order 2')
ax.set_xlabel('Cell width h')
ax.set_ylabel('Discrete L² error')
ax.legend(frameon=False)
display(fig)
plt.close(fig)
''')
heading('### Temporal convergence')
code('''
temporal, rhs_difference, mean_difference = v.temporal()
display(temporal)
fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
ax.loglog(temporal.dt, temporal.RMS_error, 'o-', color=r.COLORS['CHPM'], label='IMEX–BDF2')
ax.loglog(temporal.dt, temporal.RMS_error.iloc[-1] * (temporal.dt / temporal.dt.iloc[-1]) ** 2, '--', color=r.COLORS['CH'], label='Order 2')
ax.set_xlabel('Time step')
ax.set_ylabel('Full-grid RMS error at T = 2')
ax.legend(frameon=False)
display(fig)
plt.close(fig)
''')
heading('### Mass balance and operator agreement')
code('''
mass_residual, constant_error = v.mass()
display(pd.DataFrame([
    {'Quantity': 'Maximum absolute RHS difference', 'Value': rhs_difference},
    {'Quantity': 'Absolute mean change with zero fidelity', 'Value': mean_difference},
    {'Quantity': 'Semidiscrete mean-identity residual', 'Value': mass_residual},
    {'Quantity': 'Constant-state maximum error', 'Value': constant_error},
    {'Quantity': 'Tolerance-halving L2 difference', 'Value': tolerance_difference}
]))
''')
heading('## Complete reconstruction results')
heading('### Synthetic masked Dice')
code("display(r.frame('synthetic').pivot(index='case_id', columns='method', values='masked_dice')[r.METHODS])")
heading('### Retinal source-image masked Dice')
code("display(r.frame('retina').groupby(['image_id', 'method']).masked_dice.mean().unstack()[r.METHODS])")
heading('### Retinal case-level masked Dice')
code("display(r.frame('retina').pivot(index='case_id', columns='method', values='masked_dice')[r.METHODS])")
heading('### Retinal case-level masked Dice, VK reference')
code("display(r.frame('retina', 'VK').pivot(index='case_id', columns='method', values='masked_dice')[r.METHODS])")
heading('### Case-level outcomes')
code("complete_scores = pd.concat([r.frame('synthetic'), r.frame('retina')], ignore_index=True)\ndisplay(complete_scores[['dataset', 'case_id', 'method', 'masked_dice', 'masked_cldice', 'masked_rmse', 'masked_iou', 'boundary_f1', 'outside_rmse', 'observed_fidelity_rmse', 'phase_overshoot']].rename(columns=r.NAMES))")
heading('## Synthetic reconstructions')
for family, name in [('connections', 'Connections'), ('junctions', 'Junctions and contours'), ('multiscale', 'Multiscale structures'), ('noise', 'Noisy observations')]:
    heading('### ' + name)
    code(f"ids = [c['id'] for c in cases if c['dataset'] == 'synthetic' and c['family'] == '{family}']\nfig = r.gallery('synthetic', ids)\ndisplay(fig)\nplt.close(fig)")
heading('## Retinal reconstructions')
source_ids = sorted({c['metadata']['image_id'] for c in json.loads((ROOT / 'data' / 'cases.json').read_text()) if c['dataset'] == 'retina'})
for image_id in source_ids:
    heading('### ' + image_id)
    code(f"ids = [c['id'] for c in cases if c['dataset'] == 'retina' and c['metadata']['image_id'] == '{image_id}']\nfig = r.gallery('retina', ids)\ndisplay(fig)\nplt.close(fig)")
heading('## Data\n\nSTARE: A. Hoover, V. Kouznetsova and M. Goldbaum, IEEE Transactions on Medical Imaging 19(3), 203–210 (2000). https://doi.org/10.1109/42.845178')

notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
notebook.metadata.language_info = {'name': 'python', 'version': '3.12'}

def progress(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Cell {cell_index + 1}/{len(cells)}', flush=True)

os.chdir(ROOT)
shell = InteractiveShell.instance()
for index, cell in enumerate(notebook.cells):
    if cell.cell_type != 'code':
        continue
    progress(cell, index)
    with capture_output() as captured:
        result = shell.run_cell(cell.source, store_history=True)
    if not result.success:
        raise result.error_before_exec or result.error_in_exec
    cell.execution_count = result.execution_count
    cell.outputs = [nbformat.v4.new_output('display_data', data=item.data, metadata=item.metadata) for item in captured.outputs]
    for name in ['stdout', 'stderr']:
        value = getattr(captured, name)
        if value:
            cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=value))
destination = ROOT / 'notebooks' / 'chpm_results.ipynb'
nbformat.write(notebook, destination)

renderer = mistune.create_markdown(escape=False)
body, navigation = [], []
for cell in notebook.cells:
    if cell.cell_type == 'markdown':
        source = cell.source
        title = source.splitlines()[0]
        slug = re.sub(r'[^a-z0-9 -]', '', title.lower()).strip().replace(' ', '-')
        rendered = renderer(source)
        if title.startswith('## '):
            navigation.append(f'<a href="#{slug}">{html.escape(title[3:])}</a>')
        body.append(f'<section id="{slug}" class="markdown">{rendered}</section>')
    else:
        body.append(f'<details class="code"><summary>Code</summary><pre><code>{html.escape(cell.source)}</code></pre></details>')
        for output in cell.get('outputs', []):
            if output.output_type == 'error':
                raise RuntimeError(output.evalue)
            data = output.get('data', {})
            if 'image/png' in data:
                body.append(f'<figure><img src="data:image/png;base64,{data["image/png"]}" alt="Numerical results"></figure>')
            elif 'image/svg+xml' in data:
                body.append('<figure>' + data['image/svg+xml'] + '</figure>')
            elif 'text/html' in data:
                body.append('<div class="table-output">' + data['text/html'] + '</div>')
            elif 'text/plain' in data:
                body.append('<pre class="output">' + html.escape(data['text/plain']) + '</pre>')
            elif output.output_type == 'stream':
                body.append('<pre class="output">' + html.escape(output.text) + '</pre>')

stylesheet = '''body{margin:0;background:#fafaf7;color:#20383e;font:14px/1.55 system-ui,sans-serif}header{padding:20px 32px;border-bottom:1px solid #d9e2dc;display:flex;justify-content:space-between}a{color:#167367;text-decoration:none}nav{position:fixed;left:24px;top:100px;bottom:25px;width:210px;overflow:auto}nav a{display:block;font-size:11px;margin:0 0 12px}main{margin:35px 45px 70px 275px;max-width:1200px}h1,h2{font-family:Georgia,serif;font-weight:400}h1{font-size:43px}h2{font-size:29px;margin-top:52px;border-top:1px solid #d9e2dc;padding-top:24px}h3{font-size:17px;margin-top:30px}figure{margin:20px 0 35px}figure img,figure svg{max-width:100%;height:auto}section{scroll-margin-top:20px}.code{margin:16px 0;font-size:11px;color:#526b6b}.code summary{cursor:pointer;color:#167367}.code pre{padding:16px;background:#eaf0e9;border:1px solid #d9e2dc;overflow:auto;max-height:520px;color:#20383e;font:11px/1.65 ui-monospace,monospace}.table-output{max-width:100%;overflow:auto;margin:15px 0 25px;max-height:640px}table{border-collapse:collapse;font:11px/1.6 ui-monospace,monospace;width:100%;white-space:nowrap}thead th{position:sticky;top:0;background:#edf2eb}td,th{border:0;border-bottom:1px solid #d9e2dc;padding:7px 10px!important;text-align:right}tbody tr:hover{background:#e9f0e9}.output{font-size:11px;overflow:auto}p{max-width:800px}header strong{font-weight:600}.markdown p{font-size:13px;color:#526b6b}@media(max-width:850px){nav{position:static;display:flex;flex-wrap:wrap;width:auto;gap:8px 18px;margin:25px 20px;max-height:none}nav a{margin:0;font-size:10px}main{margin:25px 20px 60px}header{padding:17px 20px;font-size:12px}h1{font-size:35px}h2{font-size:25px}.table-output{max-height:440px}}'''
page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CHPM · Numerical notebook</title><style>' + stylesheet + '</style></head><body><header><a href="../index.html">← CHPM research</a><a href="chpm_results.ipynb" download>Download executed notebook ↓</a></header><nav aria-label="Notebook sections">' + ''.join(navigation) + '</nav><main>' + ''.join(body) + '</main></body></html>'
(ROOT / 'notebooks' / 'chpm_results.html').write_text(page)
print(json.dumps({'notebook': str(destination), 'code_cells': sum(c.cell_type == 'code' for c in notebook.cells), 'executed_cells': sum(c.cell_type == 'code' and c.execution_count is not None for c in notebook.cells), 'outputs': sum(len(c.get('outputs', [])) for c in notebook.cells)}))
