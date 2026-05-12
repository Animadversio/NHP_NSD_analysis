"""NSD_N3 dataset loader — LOC array recordings, 5 macaques."""

import os
import h5py
import numpy as np

DATA_ROOT = os.environ.get(
    'NSD_N3_ROOT',
    '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3',
)

SESSIONS = {
    'JianJian':     'GoodUnit_240629_JianJian_NSD1000_LOC_g2.mat',
    'FaCai':        'GoodUnit_240711_FaCai_NSD1000_LOC_g4.mat',
    'TuTu':         'GoodUnit_240724_TuTu_NSD1000_LOC_g2.mat',
    'ZhuangZhuang': 'GoodUnit_240817_ZhuangZhuang_NSD1000_LOC_g6.mat',
    'MaoDan':       'GoodUnit_240815_MaoDan_NSD1000_LOC_g5.mat',
}


def load_session(monkey: str, data_root: str = DATA_ROOT) -> dict:
    """
    Load one NSD_N3 GoodUnit session.

    Parameters
    ----------
    monkey : str
        One of 'JianJian', 'FaCai', 'TuTu', 'ZhuangZhuang', 'MaoDan'.
    data_root : str
        Path to directory containing the .mat files.

    Returns
    -------
    dict with keys:
        'response'   : (n_units, n_time, n_images) float32
        'time_ms'    : (n_time,) float64  — PsthRange in ms
        'n_units'    : int
        'spikepos'   : (n_units, 2)       — electrode position
        'monkey'     : str
    """
    from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
    fname = SESSIONS[monkey]
    fpath = os.path.join(data_root, fname)
    fh = h5py.File(fpath, 'r')
    d  = load_data_from_GoodUnitStrc(fh)
    out = {
        'response':  d['response_matrix_img'].astype(np.float32),  # (n_units, n_time, n_images)
        'time_ms':   d['PsthRange'],
        'n_units':   d['response_matrix_img'].shape[0],
        'spikepos':  d.get('spikepos', None),
        'monkey':    monkey,
    }
    fh.close()
    return out


def time_indices(time_ms: np.ndarray, t_start: float = -49, stride: int = 5) -> np.ndarray:
    """Return indices into time_ms for strided time bins starting at t_start."""
    return np.where(
        (time_ms >= t_start) & (np.arange(len(time_ms)) % stride == 0)
    )[0]
