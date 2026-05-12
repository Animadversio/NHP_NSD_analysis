"""Triple-N dataset loader — V1/V2/V4/IT recordings, 90 sessions, 5 macaques."""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio

TRIPLE_N_ROOT = os.environ.get(
    'TRIPLE_N_ROOT',
    '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/Triple_N',
)

# Cached metadata (loaded once on first use)
_area_df:  pd.DataFrame | None = None
_area_xyz: pd.DataFrame | None = None


def area_metadata(root: str = TRIPLE_N_ROOT) -> pd.DataFrame:
    """Return exclude_area.xls as a DataFrame (cached after first load)."""
    global _area_df
    if _area_df is None:
        _area_df = pd.read_excel(f'{root}/Others/exclude_area.xls', engine='xlrd')
    return _area_df


def area_xyz(root: str = TRIPLE_N_ROOT) -> pd.DataFrame:
    """Return AreaXYZ.xlsx as a DataFrame (cached after first load)."""
    global _area_xyz
    if _area_xyz is None:
        _area_xyz = pd.read_excel(f'{root}/Others/AreaXYZ.xlsx', engine='openpyxl')
    return _area_xyz


def load_session(ses_idx: int, root: str = TRIPLE_N_ROOT) -> dict | None:
    """
    Load Processed_sesXX_*.mat for a given session index.

    Returns
    -------
    dict with keys:
        'response'      : (n_units, 1072) float32  — mean firing rate per image
        'reliability'   : (n_units,) float32        — split-half Spearman-Brown
        'pos'           : (n_units,) float32         — electrode depth (µm)
        'UnitType'      : (n_units,) int             — 1=SUA, 2=MUA, 3=non-somatic
        'B_SI','F_SI','O_SI': (n_units,)              — body/face/object d-prime
        'mean_psth'     : (n_units, 451) or None
        'fname'         : str
        'ses_idx'       : int
    Returns None if session file not found.
    """
    files = glob.glob(f'{root}/Processed/Processed_ses{ses_idx:02d}_*.mat')
    if not files:
        return None
    fpath = files[0]
    d = sio.loadmat(fpath)
    return {
        'fname':      os.path.basename(fpath),
        'ses_idx':    ses_idx,
        'response':   d['response_best'].astype(np.float32),
        'reliability': np.array(d['reliability_best']).ravel().astype(np.float32),
        'pos':        np.array(d['pos']).ravel().astype(np.float32),
        'UnitType':   np.array(d['UnitType']).ravel().astype(int),
        'B_SI':       np.array(d['B_SI']).ravel().astype(np.float32),
        'F_SI':       np.array(d['F_SI']).ravel().astype(np.float32),
        'O_SI':       np.array(d['O_SI']).ravel().astype(np.float32),
        'mean_psth':  d.get('mean_psth', None),
    }


def load_goodunit(ses_idx: int, root: str = TRIPLE_N_ROOT):
    """
    Load GoodUnit_*.mat (PSTH data) for a given session index via h5py.
    Returns the raw h5py File handle — caller must close it.
    """
    import h5py
    files = glob.glob(f'{root}/GoodUnit/GoodUnit_*_ses{ses_idx:02d}_*.mat')
    # Fallback: match by date embedded in filename via Processed file
    if not files:
        proc_files = glob.glob(f'{root}/Processed/Processed_ses{ses_idx:02d}_*.mat')
        if proc_files:
            # e.g. Processed_ses01_240629_M1_2.mat → date=240629, monkey=M1
            parts = os.path.basename(proc_files[0]).replace('.mat','').split('_')
            date, subj = parts[2], parts[3]
            files = glob.glob(f'{root}/GoodUnit/GoodUnit_{date}_*_NSD1000_*_{subj[1:]}*.mat')
    if not files:
        return None
    return h5py.File(files[0], 'r')


def get_area_mask(ses_idx: int, pos: np.ndarray, arealabel: str,
                  root: str = TRIPLE_N_ROOT) -> np.ndarray:
    """
    Boolean mask selecting units within the depth range of arealabel for this session.

    Parameters
    ----------
    arealabel : str
        e.g. 'V1', 'V4', 'MB1'. Use 'IT' or 'EVC' for macro-area.
    """
    df = area_metadata(root)
    if arealabel in ('IT', 'EVC'):
        rows = df[(df['SesIdx'] == ses_idx) & (df['Area'] == arealabel)]
    else:
        rows = df[(df['SesIdx'] == ses_idx) & (df['AREALABEL'] == arealabel)]
    mask = np.zeros(len(pos), dtype=bool)
    for _, row in rows.iterrows():
        mask |= (pos >= row['y1']) & (pos <= row['y2'])
    return mask


def extract_area_units(
    arealabel: str,
    reliability_threshold: float = 0.2,
    unit_type: int | None = None,
    root: str = TRIPLE_N_ROOT,
) -> dict:
    """
    Extract all units from a given brain area across all 90 sessions.

    Parameters
    ----------
    arealabel : str
        Area label from exclude_area.xls, e.g. 'V1', 'V2', 'V4',
        'MB1', 'MF1', 'IT' (all IT), 'EVC' (all V1/V2/V4).
    reliability_threshold : float
        Minimum split-half reliability (default 0.2).
    unit_type : int or None
        Filter by UnitType: 1=SUA, 2=MUA. None = keep all.

    Returns
    -------
    dict with keys:
        'response'    : (N, 1072) float32
        'reliability' : (N,) float32
        'pos'         : (N,) float32
        'ses_idx'     : (N,) int
        'unit_idx'    : (N,) int        — within-session index
        'arealabel'   : str
    """
    df = area_metadata(root)
    if arealabel in ('IT', 'EVC'):
        relevant_ses = df[df['Area'] == arealabel]['SesIdx'].unique()
    else:
        relevant_ses = df[df['AREALABEL'] == arealabel]['SesIdx'].unique()

    resp_list, rel_list, pos_list, ses_list, uidx_list = [], [], [], [], []

    for ses_idx in sorted(relevant_ses):
        d = load_session(ses_idx, root)
        if d is None:
            continue
        area_mask = get_area_mask(ses_idx, d['pos'], arealabel, root)
        qual_mask = d['reliability'] >= reliability_threshold
        if unit_type is not None:
            qual_mask &= d['UnitType'] == unit_type
        mask = area_mask & qual_mask
        if not mask.any():
            continue
        resp_list.append(d['response'][mask])
        rel_list.append(d['reliability'][mask])
        pos_list.append(d['pos'][mask])
        ses_list.append(np.full(mask.sum(), ses_idx, dtype=int))
        uidx_list.append(np.where(mask)[0])

    if not resp_list:
        return {}
    return {
        'response':    np.concatenate(resp_list),
        'reliability': np.concatenate(rel_list),
        'pos':         np.concatenate(pos_list),
        'ses_idx':     np.concatenate(ses_list),
        'unit_idx':    np.concatenate(uidx_list),
        'arealabel':   arealabel,
    }


def session_summary(root: str = TRIPLE_N_ROOT) -> pd.DataFrame:
    """
    Build a DataFrame summarising all 90 sessions:
    ses_idx, fname, n_units, n_reliable, mean_rel, areas, macro_area.
    """
    df_meta = area_metadata(root)
    records = []
    for ses_idx in range(1, 91):
        d = load_session(ses_idx, root)
        if d is None:
            continue
        area_rows  = df_meta[df_meta['SesIdx'] == ses_idx]
        areas      = ','.join(area_rows['AREALABEL'].unique()) if len(area_rows) else 'Unknown'
        macro_area = area_rows['Area'].iloc[0] if len(area_rows) else 'Unknown'
        rel = d['reliability']
        records.append({
            'ses_idx':    ses_idx,
            'fname':      d['fname'],
            'n_units':    d['response'].shape[0],
            'n_reliable': int((rel >= 0.2).sum()),
            'mean_rel':   float(rel[rel > 0].mean()) if (rel > 0).any() else 0.0,
            'areas':      areas,
            'macro_area': macro_area,
        })
    return pd.DataFrame(records)
