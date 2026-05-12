"""
NSD N3 Dataset Exploration
Atlas agent — Apr 7, 2026
Generates figures and compiles into a Jupyter notebook.
"""
import sys, os, glob, json
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import basename, join
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import PIL.Image

DATA_ROOT = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3'
OUT_DIR   = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks'
FIG_DIR   = join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ─── helpers ──────────────────────────────────────────────────────────────────
def load_session(fp):
    f = h5py.File(fp, 'r')
    d = load_data_from_GoodUnitStrc(f)
    f.close()
    bn = basename(fp).replace('.mat','')
    parts = bn.split('_')
    d['date'] = parts[1]; d['monkey'] = parts[2]; d['fname'] = bn
    return d

def evk_fr(d, t0=100, t1=250):
    """Mean evoked firing rate (sp/s) per unit per image."""
    psth = d['PsthRange']
    mask = (psth >= t0) & (psth <= t1)
    R = d['response_matrix_img']        # (units, time, images)
    dt = (psth[1] - psth[0]) * 1e-3    # bin width in seconds
    return R[:, mask, :].mean(axis=1) / dt   # sp/s

def bsl_fr(d, t0=-40, t1=0):
    psth = d['PsthRange']
    mask = (psth >= t0) & (psth <= t1)
    R = d['response_matrix_img']
    dt = (psth[1] - psth[0]) * 1e-3
    return R[:, mask, :].mean(axis=1) / dt

# ─── 1. Scan all sessions ─────────────────────────────────────────────────────
print("Scanning sessions...")
files = sorted(glob.glob(join(DATA_ROOT, 'GoodUnit_*.mat')))
sessions_meta = []
for fp in files:
    bn = basename(fp).replace('.mat','')
    parts = bn.split('_')
    date, monkey = parts[1], parts[2]
    f = h5py.File(fp, 'r')
    d = load_data_from_GoodUnitStrc(f)
    f.close()
    ut = d['unittype']
    sessions_meta.append({
        'date': date, 'monkey': monkey, 'n_units': d['n_units'],
        'n_images': d['n_images'], 'n_trials': d['n_valid_trials'],
        'n_su': int((ut == 1).sum()), 'n_mu': int((ut > 1).sum()),
        'fp': fp
    })
    print(f"  {date} {monkey}: {d['n_units']} units")

monkeys = sorted(set(s['monkey'] for s in sessions_meta))
monkey_colors = {'JianJian':'#1f77b4','FaCai':'#ff7f0e','TuTu':'#2ca02c',
                 'MaoDan':'#d62728','ZhuangZhuang':'#9467bd'}
print(f"\n{len(sessions_meta)} sessions | {len(monkeys)} monkeys: {monkeys}")

# ─── FIG 1: Dataset overview ──────────────────────────────────────────────────
print("\n[Fig 1] Dataset overview...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('NSD-N3 Dataset Overview', fontsize=14, fontweight='bold')

# Units per session
ax = axes[0]
for s in sessions_meta:
    c = monkey_colors.get(s['monkey'], 'gray')
    ax.bar(s['date']+'_'+s['monkey'][:2], s['n_units'], color=c, alpha=0.8)
ax.set_xlabel('Session'); ax.set_ylabel('# Units')
ax.set_title('Units per Session')
ax.tick_params(axis='x', rotation=90, labelsize=5)
handles = [plt.Rectangle((0,0),1,1,fc=monkey_colors.get(m,'gray')) for m in monkeys]
ax.legend(handles, monkeys, fontsize=7, loc='upper left')

# Units per monkey
ax = axes[1]
monkey_units = {m: sum(s['n_units'] for s in sessions_meta if s['monkey']==m) for m in monkeys}
bars = ax.bar(monkey_units.keys(), monkey_units.values(),
              color=[monkey_colors.get(m,'gray') for m in monkey_units])
ax.set_title('Total Units per Monkey'); ax.set_ylabel('# Units')
ax.set_xlabel('Monkey'); ax.tick_params(axis='x', rotation=20)
for bar, v in zip(bars, monkey_units.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, str(v), ha='center', fontsize=8)

# Sessions per monkey
ax = axes[2]
monkey_sess = {m: sum(1 for s in sessions_meta if s['monkey']==m) for m in monkeys}
bars = ax.bar(monkey_sess.keys(), monkey_sess.values(),
              color=[monkey_colors.get(m,'gray') for m in monkey_sess])
ax.set_title('Sessions per Monkey'); ax.set_ylabel('# Sessions')
ax.set_xlabel('Monkey'); ax.tick_params(axis='x', rotation=20)
for bar, v in zip(bars, monkey_sess.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, str(v), ha='center', fontsize=9)

plt.tight_layout()
fig1_path = join(FIG_DIR, 'fig1_dataset_overview.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {fig1_path}")

# ─── FIG 2: Neural responses from one session ─────────────────────────────────
print("\n[Fig 2] Single-session neural responses (JianJian 240629)...")
d = load_session(files[0])  # JianJian 240629
psth = d['PsthRange']
R = d['response_matrix_img']  # (units, time, images)

# Mean PSTH across all units and images
dt = (psth[1]-psth[0])*1e-3
mean_psth = R.mean(axis=(0,2)) / dt  # sp/s averaged over units & images

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Neural Responses — JianJian 240629 ({d["n_units"]} units, {d["n_images"]} images)',
             fontsize=13, fontweight='bold')

# Mean population PSTH
ax = axes[0,0]
ax.axvspan(100, 250, alpha=0.15, color='orange', label='evoked window')
ax.axvspan(-40, 0, alpha=0.15, color='blue', label='baseline')
ax.axvline(0, color='k', lw=1, ls='--')
ax.plot(psth, mean_psth, color='steelblue', lw=1.5)
ax.set_xlabel('Time from stimulus onset (ms)')
ax.set_ylabel('Firing rate (sp/s)')
ax.set_title('Mean Population PSTH')
ax.legend(fontsize=8)

# Distribution of evoked firing rates
ax = axes[0,1]
evk = evk_fr(d)   # (units, images)
bsl = bsl_fr(d)
unit_mean_evk = evk.mean(axis=1)
unit_mean_bsl = bsl.mean(axis=1)
ax.hist(unit_mean_bsl, bins=40, alpha=0.6, label='Baseline', color='royalblue')
ax.hist(unit_mean_evk, bins=40, alpha=0.6, label='Evoked (100-250ms)', color='tomato')
ax.set_xlabel('Mean firing rate (sp/s)'); ax.set_ylabel('# Units')
ax.set_title('Distribution of Unit Firing Rates')
ax.legend()
ax.set_xlim(-5, np.percentile(unit_mean_evk, 99))

# Response matrix heatmap (top 50 responsive units, first 100 images)
ax = axes[1,0]
selectivity = evk.std(axis=1)  # variability across images
top_idx = np.argsort(selectivity)[-50:][::-1]
mat = evk[top_idx, :100]
mat_z = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-6)
im = ax.imshow(mat_z, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax.set_xlabel('Image index (first 100)'); ax.set_ylabel('Unit (sorted by selectivity)')
ax.set_title('Response Matrix — top 50 selective units (z-scored)')
plt.colorbar(im, ax=ax, label='z-score')

# Unit type breakdown
ax = axes[1,1]
unittype_names = {1:'SUA', 2:'MUA-A', 3:'MUA-B', 4:'MUA-C'}
ut_counts = {unittype_names.get(int(u), str(int(u))): int((d['unittype']==u).sum())
             for u in np.unique(d['unittype'])}
bars = ax.bar(ut_counts.keys(), ut_counts.values(), color=['gold','steelblue','tomato','green'])
ax.set_title('Unit Type Composition'); ax.set_ylabel('# Units'); ax.set_xlabel('Unit Type')
for bar, v in zip(bars, ut_counts.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(v), ha='center', fontsize=9)

plt.tight_layout()
fig2_path = join(FIG_DIR, 'fig2_single_session_responses.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {fig2_path}")

# ─── FIG 3: Stimulus images sample ────────────────────────────────────────────
print("\n[Fig 3] Sample stimulus images...")
img_dir = join(DATA_ROOT, 'NSD1000_LOC')
img_files = sorted(glob.glob(join(img_dir, '*.bmp')))

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('NSD-N3 Stimulus Sample (32 of 1072 images)', fontsize=13, fontweight='bold')
np.random.seed(42)
sample_idx = np.random.choice(len(img_files), 32, replace=False)
sample_idx.sort()
for i, ax in enumerate(axes.flat):
    img = PIL.Image.open(img_files[sample_idx[i]])
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'#{sample_idx[i]+1}', fontsize=7)
plt.tight_layout()
fig3_path = join(FIG_DIR, 'fig3_stimulus_sample.png')
plt.savefig(fig3_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved {fig3_path}")

# ─── FIG 4: Cross-session comparison ──────────────────────────────────────────
print("\n[Fig 4] Cross-session mean firing rates...")
# Use a subset of sessions to keep it fast
sess_sample = sessions_meta[:20]
cross_data = []
for s in sess_sample:
    d = load_session(s['fp'])
    evk = evk_fr(d).mean()
    bsl = bsl_fr(d).mean()
    cross_data.append({'monkey': s['monkey'], 'date': s['date'], 'evk': evk, 'bsl': bsl,
                       'n_units': s['n_units']})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Cross-Session Neural Activity (first 20 sessions)', fontsize=13, fontweight='bold')

ax = axes[0]
for cd in cross_data:
    c = monkey_colors.get(cd['monkey'], 'gray')
    ax.scatter(cd['bsl'], cd['evk'], color=c, s=80, alpha=0.8,
               label=cd['monkey'])
ax.plot([0, 30], [0, 30], 'k--', alpha=0.4, lw=1)
ax.set_xlabel('Mean Baseline FR (sp/s)'); ax.set_ylabel('Mean Evoked FR (sp/s)')
ax.set_title('Baseline vs Evoked Firing Rate per Session')
# deduplicate legend
seen = set()
handles, labels = [], []
for h, l in zip(*ax.get_legend_handles_labels()):
    if l not in seen:
        handles.append(h); labels.append(l); seen.add(l)
ax.legend(handles, labels, fontsize=8)

ax = axes[1]
labels_x = [f"{cd['date'][-4:]}\n{cd['monkey'][:3]}" for cd in cross_data]
x = np.arange(len(cross_data))
w = 0.35
bsl_vals = [cd['bsl'] for cd in cross_data]
evk_vals = [cd['evk'] for cd in cross_data]
colors_x = [monkey_colors.get(cd['monkey'],'gray') for cd in cross_data]
bars1 = ax.bar(x - w/2, bsl_vals, w, label='Baseline', color='royalblue', alpha=0.7)
bars2 = ax.bar(x + w/2, evk_vals, w, label='Evoked', color='tomato', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels_x, fontsize=7)
ax.set_ylabel('Mean FR (sp/s)'); ax.set_title('Firing Rates per Session')
ax.legend()
plt.tight_layout()
fig4_path = join(FIG_DIR, 'fig4_cross_session.png')
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {fig4_path}")

# ─── FIG 5: Best/worst image responses for top unit ───────────────────────────
print("\n[Fig 5] Best/worst images for most selective unit...")
d = load_session(files[0])
evk = evk_fr(d)  # (units, images)
selectivity = evk.std(axis=1)
best_unit_idx = int(np.argmax(selectivity))
unit_responses = evk[best_unit_idx, :]  # (n_images,)

top10_imgs = np.argsort(unit_responses)[-10:][::-1]
bot10_imgs = np.argsort(unit_responses)[:10]

fig, axes = plt.subplots(3, 10, figsize=(20, 7))
fig.suptitle(f'Most selective unit (idx {best_unit_idx}) — Best vs Worst images\n'
             f'Selectivity (std): {selectivity[best_unit_idx]:.1f} sp/s', fontsize=11, fontweight='bold')

for col, img_idx in enumerate(top10_imgs):
    img = PIL.Image.open(join(img_dir, f'{img_idx+1:04d}.bmp'))
    axes[0, col].imshow(img); axes[0, col].axis('off')
    axes[0, col].set_title(f'#{img_idx+1}\n{unit_responses[img_idx]:.0f}sp/s', fontsize=7)
axes[0, 0].set_ylabel('Best', fontsize=9)

for col, img_idx in enumerate(bot10_imgs):
    img = PIL.Image.open(join(img_dir, f'{img_idx+1:04d}.bmp'))
    axes[1, col].imshow(img); axes[1, col].axis('off')
    axes[1, col].set_title(f'#{img_idx+1}\n{unit_responses[img_idx]:.0f}sp/s', fontsize=7)
axes[1, 0].set_ylabel('Worst', fontsize=9)

# PSTH for best unit: best vs worst 10 images
ax = axes[2, 0]
psth = d['PsthRange']
R_unit = d['response_matrix_img'][best_unit_idx]  # (time, images)
dt = (psth[1]-psth[0])*1e-3
ax = fig.add_axes([0.05, 0.02, 0.9, 0.25])
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
ax.axvspan(100, 250, alpha=0.12, color='orange')
psth_best = R_unit[:, top10_imgs].mean(axis=1) / dt
psth_worst = R_unit[:, bot10_imgs].mean(axis=1) / dt
ax.plot(psth, psth_best, color='tomato', lw=2, label='Best 10 images')
ax.plot(psth, psth_worst, color='steelblue', lw=2, label='Worst 10 images')
ax.set_xlabel('Time (ms)'); ax.set_ylabel('FR (sp/s)')
ax.set_title(f'PSTH for most selective unit (idx {best_unit_idx})')
ax.legend(fontsize=8)
for a in axes[2]: a.axis('off')
plt.subplots_adjust(hspace=0.4, top=0.88, bottom=0.3)
fig5_path = join(FIG_DIR, 'fig5_best_worst_images.png')
plt.savefig(fig5_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved {fig5_path}")

# ─── FIG 6: Electrode array layout ────────────────────────────────────────────
print("\n[Fig 6] Electrode/unit spatial layout...")
d = load_session(files[0])
evk = evk_fr(d)
mean_evk_per_unit = evk.mean(axis=1)
pos = d['spikepos']   # (units, 2) — electrode positions

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Electrode Layout — JianJian 240629', fontsize=13, fontweight='bold')

ax = axes[0]
sc = ax.scatter(pos[:,0], pos[:,1], c=mean_evk_per_unit, cmap='hot', s=20, alpha=0.8,
                vmin=0, vmax=np.percentile(mean_evk_per_unit, 95))
plt.colorbar(sc, ax=ax, label='Mean evoked FR (sp/s)')
ax.set_xlabel('X position (μm)'); ax.set_ylabel('Y position (μm)')
ax.set_title('Units colored by firing rate')

ax = axes[1]
selectivity = evk.std(axis=1)
sc = ax.scatter(pos[:,0], pos[:,1], c=selectivity, cmap='viridis', s=20, alpha=0.8,
                vmin=0, vmax=np.percentile(selectivity, 95))
plt.colorbar(sc, ax=ax, label='Selectivity (std sp/s)')
ax.set_xlabel('X position (μm)')
ax.set_title('Units colored by image selectivity')

plt.tight_layout()
fig6_path = join(FIG_DIR, 'fig6_electrode_layout.png')
plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {fig6_path}")

# ─── Compile Jupyter Notebook ─────────────────────────────────────────────────
print("\nCompiling Jupyter notebook...")
import base64

def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

nb = new_notebook()
nb.cells = [
    new_markdown_cell("# NSD-N3 Dataset Exploration\n\nAtlas agent — Apr 7, 2026\n\n"
                      "Explores the NHP NSD dataset recorded at Kempner Institute. "
                      "59 sessions, 5 monkeys (JianJian, FaCai, TuTu, MaoDan, ZhuangZhuang), "
                      "recordings from visual cortex during passive viewing of 1072 NSD images."),

    new_markdown_cell("## Dataset Overview\n\n"
                      f"- **Data path**: `{DATA_ROOT}`\n"
                      f"- **Sessions**: {len(sessions_meta)}\n"
                      f"- **Monkeys**: {', '.join(monkeys)}\n"
                      f"- **Images**: 1072 per session (NSD1000_LOC subset)\n"
                      f"- **Time range**: -49 to 400 ms around stimulus onset, 1ms bins\n"
                      f"- **Unit types**: SUA (1), MUA-A (2), MUA-B (3), MUA-C (4)\n"),

    new_code_cell("# Setup\nimport sys, glob, h5py, numpy as np\nsys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')\n"
                  "from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc\n"
                  f"DATA_ROOT = '{DATA_ROOT}'\nfiles = sorted(glob.glob(DATA_ROOT + '/GoodUnit_*.mat'))\nprint(f'{{len(files)}} sessions found')"),

    new_markdown_cell("## Fig 1: Dataset Overview"),
    new_code_cell(f"from IPython.display import Image\nImage('{fig1_path}')"),

    new_markdown_cell("## Fig 2: Single Session Neural Responses (JianJian 240629)\n\n"
                      "Shows mean population PSTH, firing rate distributions, response matrix heatmap, and unit type breakdown."),
    new_code_cell(f"Image('{fig2_path}')"),

    new_markdown_cell("## Fig 3: Sample Stimulus Images (NSD1000_LOC)"),
    new_code_cell(f"Image('{fig3_path}')"),

    new_markdown_cell("## Fig 4: Cross-Session Firing Rates"),
    new_code_cell(f"Image('{fig4_path}')"),

    new_markdown_cell("## Fig 5: Most Selective Unit — Best vs Worst Images"),
    new_code_cell(f"Image('{fig5_path}')"),

    new_markdown_cell("## Fig 6: Electrode Array Spatial Layout"),
    new_code_cell(f"Image('{fig6_path}')"),

    new_markdown_cell("## Summary\n\n"
                      "- Dataset has 59 sessions across 5 NHP subjects recorded in visual cortex\n"
                      "- Each session has hundreds of units (single and multi-unit activity)\n"
                      "- Neural responses clearly driven by visual stimuli with strong evoked activity (100-250ms)\n"
                      "- Substantial unit-to-unit and image-to-image variability in firing rates\n"
                      "- `response_matrix_img` (units × time × images) is the main analysis array\n"
                      "- Use `evk_fr()` helper for evoked firing rates per unit per image\n"),
]

nb_path = join(OUT_DIR, 'NSD_N3_exploration.ipynb')
with open(nb_path, 'w') as f:
    nbformat.write(nb, f)
print(f"Saved notebook: {nb_path}")
print("\nAll figures:")
for p in [fig1_path, fig2_path, fig3_path, fig4_path, fig5_path, fig6_path]:
    print(f"  {p}")
print("\nDone!")
