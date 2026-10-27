from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from study_statistics import _seed, mean_interval, paired_signflip, holm_adjust

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['CH', 'CHPM', 'CH-width', 'Biharmonic', 'TV', 'Frozen']
FAMILIES = ['connections', 'junctions', 'multiscale', 'noise']
COLORS = {'CH': '#7c8990', 'CHPM': '#167367', 'CH-width': '#355b91', 'Biharmonic': '#aa7848', 'TV': '#b56464', 'Frozen': '#81748e'}
NAMES = {'masked_dice': 'Masked Dice', 'masked_cldice': 'Masked clDice', 'masked_rmse': 'Masked RMSE', 'masked_iou': 'Masked IoU', 'boundary_f1': 'Boundary F1', 'outside_rmse': 'Reference RMSE', 'observed_fidelity_rmse': 'Observation RMSE', 'phase_overshoot': 'Phase overshoot'}

def read(name):
    return json.loads((ROOT / 'data' / 'records' / name).read_text())

def records(dataset):
    return read(f'{dataset}_test_rows.json')

def frame(dataset, reference='AH'):
    key = 'vk_scores' if reference == 'VK' else 'scores'
    rows = []
    for row in records(dataset):
        rows.append(dict(dataset=dataset, case_id=row['case_id'], method=row['method'], family=row['family'], image_id=row['metadata'].get('image_id', row['case_id']), width=row['metadata'].get('width', np.nan), **row[key]))
    return pd.DataFrame(rows)

def means(dataset, reference='AH', width=None):
    df = frame(dataset, reference)
    if width is not None:
        df = df[df.width == width]
    columns = [c for c in NAMES if c in df]
    if dataset == 'retina':
        df = df.groupby(['method', 'image_id'])[columns].mean().reset_index()
    return df.groupby('method')[columns].mean().reindex(METHODS)

def family_means():
    return frame('synthetic').groupby(['family', 'method'])[['masked_dice', 'masked_cldice', 'masked_rmse']].mean()

def inference(dataset, reference='AH', width=None):
    df = frame(dataset, reference)
    if width is not None:
        df = df[df.width == width]
    unit = 'image_id' if dataset == 'retina' else 'case_id'
    paired = df.groupby([unit, 'method']).masked_dice.mean().unstack().sort_index()
    score_key = 'vk_scores' if reference == 'VK' else 'scores'
    rows = []
    probabilities = []
    for comparator in ['CH', 'CH-width', 'Biharmonic', 'TV', 'Frozen']:
        difference = (paired.CHPM - paired[comparator]).to_numpy()
        seed = _seed(20261026, f'difference|{score_key}|masked_dice|CHPM|{comparator}')
        ci = mean_interval(difference, bootstrap=20000, seed=seed)
        rows.append({'Comparator': comparator, 'Difference': ci['mean'], '95% lower': ci['ci_low'], '95% upper': ci['ci_high']})
        if dataset == 'retina':
            probabilities.append(paired_signflip(difference)['p_value'])
    table = pd.DataFrame(rows)
    if probabilities:
        table['Holm p'] = holm_adjust(probabilities)
    return table

def diagnostics():
    rows = []
    for dataset in ['synthetic', 'retina']:
        for method in METHODS:
            group = [r for r in records(dataset) if r['method'] == method]
            rows.append({'Dataset': dataset, 'Method': method, 'Reference RMSE': np.mean([r['scores']['outside_rmse'] for r in group]), 'Observation RMSE': np.mean([r['scores']['observed_fidelity_rmse'] for r in group]), 'Maximum overshoot': max(r['scores']['phase_overshoot'] for r in group)})
    return pd.DataFrame(rows)

def tv_accuracy():
    rows = []
    for dataset in ['synthetic', 'retina']:
        group = [r['diagnostics'] for r in records(dataset) if r['method'] == 'TV']
        rows.append({'Dataset': dataset, 'Reconstructions': len(group), 'At tolerance': sum(r['converged'] for r in group), 'At iteration limit': sum(not r['converged'] for r in group), 'Maximum relative gap': max(r['relative_primal_dual_gap'] for r in group), 'Maximum divergence residual': max(r['dual_feasibility'] for r in group)})
    return pd.DataFrame(rows)

def selected_parameters():
    rows = []
    for dataset in ['synthetic', 'retina']:
        for method, record in read(f'{dataset}_selected_parameters.json')['selected'].items():
            p = record['parameters']
            rows.append({'Dataset': dataset, 'Method': method, **{k: p[k] for k in ['tau', 'gamma', 'K', 'T', 'dt', 'epsilon', 'fidelity']}})
    return pd.DataFrame(rows)

def tuning_table():
    rows = []
    for dataset in ['synthetic', 'retina']:
        candidates = read(f'{dataset}_tuning_summary.json')
        for method in ['CH', 'CHPM']:
            ordered = sorted([r for r in candidates if r['method'] == method], key=lambda r: (-r['mean_dice'], r['mean_rmse']))
            for rank, record in enumerate(ordered, 1):
                p = record['parameters']
                rows.append({'Dataset': dataset, 'Method': method, 'Rank': rank, 'tau': p['tau'], 'gamma': p['gamma'], 'K': p['K'], 'T': record['T'], 'Dice': record['mean_dice'], 'RMSE': record['mean_rmse']})
    return pd.DataFrame(rows)

def refinement_table(dataset, kind='time'):
    comparisons = read(f'{dataset}_summary_refinement.json')['comparisons']
    return pd.DataFrame([r for r in comparisons if r['kind'] == kind])

def case_data(case_id):
    return dict(np.load(ROOT / 'data' / 'cases' / f'{case_id}.npz'))

def field(dataset, case_id, method):
    return np.load(ROOT / 'data' / 'fields' / dataset / method / f'{case_id}.npz')['u']

def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#243c43', 'text.color': '#243c43', 'xtick.color': '#526269', 'ytick.color': '#526269', 'axes.edgecolor': '#c8d1d2', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'none'})

def score_figure():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    for ax, dataset in zip(axes, ['synthetic', 'retina']):
        y = means(dataset)['masked_dice']
        ax.barh(METHODS[::-1], y.iloc[::-1], color=[COLORS[m] for m in METHODS[::-1]], height=.62)
        for i, v in enumerate(y.iloc[::-1]):
            ax.text(v + .015, i, f'{v:.4f}', va='center', fontsize=9)
        ax.set_xlim(0, 1.06)
        ax.set_xlabel('Mean masked Dice')
        ax.set_title('Synthetic' if dataset == 'synthetic' else 'Retinal vessel masks', loc='left')
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    return fig

def family_figure():
    fig, ax = plt.subplots(figsize=(10, 3.7), constrained_layout=True)
    data = family_means()
    x = np.arange(4)
    for offset, method in [(-.18, 'CH'), (.18, 'CHPM')]:
        values = [data.loc[(family, method), 'masked_dice'] for family in FAMILIES]
        ax.bar(x + offset, values, width=.32, color=COLORS[method], label=method)
        for xx, y in zip(x + offset, values):
            ax.text(xx, y + .02, f'{y:.4f}', ha='center', fontsize=9)
    ax.set_xticks(x, ['Connections', 'Junctions', 'Multiscale', 'Noisy observations'])
    ax.set_ylim(0, 1.09)
    ax.set_ylabel('Mean masked Dice')
    ax.legend(frameon=False, ncol=2, loc='lower left')
    return fig

def retinal_figure(reference='AH'):
    data = frame('retina', reference).groupby(['image_id', 'method']).masked_dice.mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax = axes[0]
    for i, (_, row) in enumerate(data.iterrows()):
        ax.plot([row.CH, row.CHPM], [i, i], color='#ccd4d5')
    for method in ['CH', 'CHPM']:
        ax.scatter(data[method], np.arange(len(data)), color=COLORS[method], label=method, s=25, zorder=3)
    ax.set_yticks(range(len(data)), data.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Mean masked Dice')
    ax.set_title(f'{reference} reference', loc='left')
    ax.legend(frameon=False, ncol=2, loc='lower left')
    for j, method in enumerate(METHODS):
        values = [means('retina', reference, width).loc[method, 'masked_dice'] for width in [4, 8]]
        axes[1].plot(np.arange(2) + (j - 2.5) * .09, values, 'o', color=COLORS[method], label=method)
    axes[1].set_xticks([0, 1], ['4 pixels', '8 pixels'])
    axes[1].set_xlim(-.5, 1.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Mean masked Dice')
    axes[1].set_title('Scratch width', loc='left')
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc='lower left')
    return fig

def calibration_figure():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for ax, dataset in zip(axes, ['synthetic', 'retina']):
        values = np.load(ROOT / 'data' / 'records' / f'{dataset}_width_calibration_profiles.npz')
        ax.plot(values['x'], values['target'], color=COLORS['CHPM'], label='CHPM')
        ax.plot(values['x'], values['matched'], '--', color=COLORS['CH-width'], label='CH-width')
        ax.set_xlim(-7, 7)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(dataset.title(), loc='left')
        ax.set_xlabel('Distance from interface (pixels)')
        ax.set_ylabel('Phase value')
        ax.legend(frameon=False, fontsize=8)
    for j, method in enumerate(['CHPM', 'CH-width', 'Frozen']):
        values = [means(dataset).loc[method, 'masked_dice'] for dataset in ['synthetic', 'retina']]
        axes[2].bar(np.arange(2) + (j - 1) * .23, values, width=.2, color=COLORS[method], label=method)
    axes[2].set_xticks([0, 1], ['Synthetic', 'Retinal'])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('Mean masked Dice')
    axes[2].legend(frameon=False, fontsize=8, loc='lower left')
    return fig

def gallery(dataset, case_ids):
    lookup = {(r['case_id'], r['method']): r for r in records(dataset)}
    columns = ['Observed', 'Reference'] + METHODS
    fig, axes = plt.subplots(len(case_ids), 8, figsize=(14, len(case_ids) * 1.85), squeeze=False)
    for i, case_id in enumerate(case_ids):
        case = case_data(case_id)
        arrays = [case['observed'], 2 * case['truth'].astype(float) - 1] + [2 * (field(dataset, case_id, m) > 0).astype(float) - 1 for m in METHODS]
        for j, array in enumerate(arrays):
            ax = axes[i, j]
            ax.imshow(array, cmap='gray_r', vmin=-1, vmax=1, interpolation='nearest')
            ax.contour(case['mask'], levels=[.5], colors=['#42a69b'], linewidths=.45)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(columns[j], fontsize=9)
            if j >= 2:
                ax.set_xlabel(f"{lookup[(case_id, METHODS[j-2])]['scores']['masked_dice']:.4f}", fontsize=9)
        axes[i, 0].set_ylabel(case_id.replace('test_retina_', '').replace('test_', ''), fontsize=8)
    fig.tight_layout(pad=.7)
    return fig
