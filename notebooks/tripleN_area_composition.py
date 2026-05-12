"""Triple-N dataset: area composition analysis and unit extraction by brain region.

Loads all 90 Processed_sesXX_*.mat files, joins with exclude_area.xls to map
units to brain area labels (V1/V2/V4/IT patches), and generates an overview figure.

Outputs
-------
- figures/fig_tripleN_area_overview.png  — summary figure
- Prints per-area unit counts and session index tables

Usage
-----
    python tripleN_area_composition.py

To extract units for a specific area (e.g. V4):
    units = extract_area_units('V4', reliability_threshold=0.2)
    # returns dict: {ses_idx: {'response': (n_units, 1072), 'reliability': ..., 'pos': ...}}
"""

import os, glob
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

TRIPLE_N  = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/Triple_N'
FIG_DIR   = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load area metadata ────────────────────────────────────────────────────────
area_df  = pd.read_excel(f'{TRIPLE_N}/Others/exclude_area.xls',  engine='xlrd')
area_xyz = pd.read_excel(f'{TRIPLE_N}/Others/AreaXYZ.xlsx',      engine='openpyxl')

# ── Session summary ───────────────────────────────────────────────────────────
def load_session_summary(ses_idx: int) -> dict | None:
    """Load lightweight summary from Processed_sesXX_*.mat."""
    files = glob.glob(f'{TRIPLE_N}/Processed/Processed_ses{ses_idx:02d}_*.mat')
    if not files:
        return None
    d = sio.loadmat(files[0])
    return {
        'fname':       os.path.basename(files[0]),
        'n_units':     d['response_best'].shape[0],
        'response':    d['response_best'],           # (n_units, 1072)
        'reliability': np.array(d['reliability_best']).ravel(),
        'pos':         np.array(d['pos']).ravel(),   # depth along shank (µm)
        'UnitType':    np.array(d['UnitType']).ravel(),
        'B_SI':        np.array(d['B_SI']).ravel(),
        'F_SI':        np.array(d['F_SI']).ravel(),
        'O_SI':        np.array(d['O_SI']).ravel(),
        'mean_psth':   d.get('mean_psth', None),     # (n_units, 451) if present
    }


def get_unit_area_mask(ses_idx: int, pos: np.ndarray, arealabel: str) -> np.ndarray:
    """Boolean mask of units in a given AREALABEL for this session."""
    rows = area_df[(area_df['SesIdx'] == ses_idx) & (area_df['AREALABEL'] == arealabel)]
    mask = np.zeros(len(pos), dtype=bool)
    for _, row in rows.iterrows():
        mask |= (pos >= row['y1']) & (pos <= row['y2'])
    return mask


def extract_area_units(arealabel: str, reliability_threshold: float = 0.2) -> dict:
    """
    Extract all units from a given AREALABEL across all sessions.

    Parameters
    ----------
    arealabel : str
        e.g. 'V1', 'V2', 'V4', 'MB1', 'MB2', 'MF1' — see exclude_area.xls
        Use 'IT' or 'EVC' for macro-area grouping.
    reliability_threshold : float
        Minimum split-half reliability to include a unit.

    Returns
    -------
    dict with keys:
        'response'    : (n_units_total, 1072) float32
        'reliability' : (n_units_total,) float32
        'pos'         : (n_units_total,) float32  — depth in µm
        'ses_idx'     : (n_units_total,) int       — session of origin
        'unit_idx'    : (n_units_total,) int       — within-session unit index
    """
    # Determine which sessions to look in
    if arealabel in ('IT', 'EVC'):
        relevant = area_df[area_df['Area'] == arealabel]['SesIdx'].unique()
    else:
        relevant = area_df[area_df['AREALABEL'] == arealabel]['SesIdx'].unique()

    resp_list, rel_list, pos_list, ses_list, uidx_list = [], [], [], [], []

    for ses_idx in sorted(relevant):
        d = load_session_summary(ses_idx)
        if d is None:
            continue
        pos = d['pos']
        rel = d['reliability']

        if arealabel in ('IT', 'EVC'):
            macro = area_df[area_df['SesIdx'] == ses_idx]['Area'].iloc[0]
            if macro != arealabel:
                continue
            area_rows = area_df[area_df['SesIdx'] == ses_idx]
            area_mask = np.zeros(len(pos), dtype=bool)
            for _, row in area_rows.iterrows():
                area_mask |= (pos >= row['y1']) & (pos <= row['y2'])
        else:
            area_mask = get_unit_area_mask(ses_idx, pos, arealabel)

        qual_mask = rel >= reliability_threshold
        mask = area_mask & qual_mask
        if not mask.any():
            continue

        resp_list.append(d['response'][mask].astype(np.float32))
        rel_list.append(rel[mask])
        pos_list.append(pos[mask])
        ses_list.append(np.full(mask.sum(), ses_idx, dtype=int))
        uidx_list.append(np.where(mask)[0])

    if not resp_list:
        return {}

    return {
        'response':    np.concatenate(resp_list, axis=0),
        'reliability': np.concatenate(rel_list),
        'pos':         np.concatenate(pos_list),
        'ses_idx':     np.concatenate(ses_list),
        'unit_idx':    np.concatenate(uidx_list),
    }


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_area_overview(save: bool = True):
    records = []
    for ses_idx in range(1, 91):
        d = load_session_summary(ses_idx)
        if d is None:
            continue
        area_rows = area_df[area_df['SesIdx'] == ses_idx]
        areas      = ','.join(area_rows['AREALABEL'].unique()) if len(area_rows) else 'Unknown'
        macro_area = area_rows['Area'].iloc[0] if len(area_rows) else 'Unknown'
        records.append({
            'ses': ses_idx,
            'n_units':    d['n_units'],
            'n_reliable': int((d['reliability'] >= 0.2).sum()),
            'mean_rel':   float(d['reliability'][d['reliability'] > 0].mean()) if (d['reliability'] > 0).any() else 0,
            'areas':      areas,
            'macro_area': macro_area,
        })
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'IT': 'steelblue', 'EVC': 'tomato', 'Unknown': 'gray'}

    # Panel 1: units per session
    ax = axes[0]
    for macro, grp in df.groupby('macro_area'):
        ax.scatter(grp['ses'], grp['n_units'],    c=colors.get(macro, 'gray'), label=macro, s=40, alpha=0.8)
        ax.scatter(grp['ses'], grp['n_reliable'], c=colors.get(macro, 'gray'), s=40, alpha=0.4, marker='^')
    ax.axvline(70.5, color='k', lw=1, ls='--', alpha=0.5)
    ax.text(71.5, ax.get_ylim()[1]*0.95, 'EVC\n(ses71+)', fontsize=8)
    ax.set_xlabel('Session'); ax.set_ylabel('Units')
    ax.set_title('Units per session\n(●=total, △=reliable≥0.2)')
    ax.legend()

    # Panel 2: pie by area
    ax = axes[1]
    evc_counts = {a: df[df['areas'].str.contains(a, na=False)]['n_reliable'].sum()
                  for a in ['V1', 'V2', 'V4']}
    it_counts_raw = df[df['macro_area'] == 'IT'].groupby('areas')['n_reliable'].sum()
    it_top = it_counts_raw.nlargest(5)
    pie_data = {**{f'IT/{k[:5]}': v for k, v in it_top.items()}, **evc_counts}
    cmap = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
    ax.pie(list(pie_data.values()), labels=list(pie_data.keys()),
           colors=cmap, autopct='%1.0f%%', startangle=90)
    it_n  = df[df['macro_area'] == 'IT']['n_reliable'].sum()
    evc_n = df[df['macro_area'] == 'EVC']['n_reliable'].sum()
    ax.set_title(f'Reliable units by area\n(IT: {it_n:,}  EVC: {evc_n:,})')

    # Panel 3: mean reliability per area label
    ax = axes[2]
    area_rel = df.groupby('areas')['mean_rel'].mean().sort_values(ascending=False).head(15)
    bar_colors = ['tomato' if any(v in a for v in ['V1','V2','V4','V12']) else 'steelblue'
                  for a in area_rel.index]
    ax.barh(range(len(area_rel)), area_rel.values, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(area_rel)))
    ax.set_yticklabels(area_rel.index, fontsize=9)
    ax.set_xlabel('Mean split-half reliability')
    ax.set_title('Reliability by area\n(red=EVC, blue=IT)')

    plt.tight_layout()
    if save:
        out = os.path.join(FIG_DIR, 'fig_tripleN_area_overview.png')
        plt.savefig(out, dpi=130, bbox_inches='tight')
        print(f'Saved: {out}')
    plt.show()
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Triple-N Area Composition ===\n')

    print('Area metadata:')
    print(f'  {len(area_df)} ROI entries across {area_df["SesIdx"].nunique()} sessions')
    print(f'  IT sessions:  {(area_df["Area"]=="IT").sum()}')
    print(f'  EVC sessions: {(area_df["Area"]=="EVC").sum()}')
    print(f'  Area labels: {sorted(area_df["AREALABEL"].unique())}\n')

    df_summary = plot_area_overview(save=True)

    print('\nPer macro-area totals:')
    print(df_summary.groupby('macro_area')[['n_units','n_reliable']].sum().to_string())

    print('\nEVC sessions:')
    print(df_summary[df_summary['macro_area']=='EVC'][
        ['ses','areas','n_units','n_reliable','mean_rel']].to_string(index=False))

    # Example: extract all V4 units
    print('\nExtracting V4 units (reliability >= 0.2)...')
    v4 = extract_area_units('V4', reliability_threshold=0.2)
    if v4:
        print(f'  V4: {v4["response"].shape[0]} units from sessions {np.unique(v4["ses_idx"])}')
        print(f'  Response shape: {v4["response"].shape}')
        print(f'  Mean reliability: {v4["reliability"].mean():.3f}')
