# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains tools and analysis code for working with the NSD (Natural Scene Dataset) from Non-Human Primate (NHP) experiments. The project focuses on loading, parsing, and analyzing neural data stored in MATLAB files, along with associated image stimuli.

## Key Dependencies

Install the required Python packages:
```bash
pip install scipy mat73 numpy pandas h5py matplotlib seaborn scikit-learn torch torchvision easydict
```

Critical dependencies:
- `scipy`: For loading older MATLAB format files (Processed files)
- `mat73`: For loading MATLAB v7.3 format files (GoodUnit files) 
- `h5py`: For HDF5/MATLAB v7.3 file parsing
- `torch` + `torchvision`: For image processing and neural network operations

## Data Structure

The project works with two main types of MATLAB files:

### GoodUnit Files
Format: `GoodUnit_YYYYMMDD_MonkeyName_NSD1000_LOC_gN.mat`
- Contains neural unit data in MATLAB v7.3 format
- Key structures: `GoodUnitStrc`, `global_params`, `meta_data`, `trial_ML`
- Loaded using `mat73` or `h5py`

### Processed Files  
Format: `Processed_sesNN_YYYYMMDD_MN_N.mat`
- Contains processed session data in older MATLAB format
- Loaded using `scipy.io.loadmat`

## Architecture

### Core Components

- **`NSD_utils/`**: Core utility modules
  - `h5_dataset_utils.py`: Functions for loading and parsing HDF5/MATLAB files
  - `image_utils.py`: PIL image grid utilities for visualization

- **`agent_out/`**: Data loading infrastructure
  - `final_nsd_loader.py`: Comprehensive NSD data loader class
  - `data_loader.py`: Alternative loader implementation
  - Contains comprehensive README with usage examples

- **`notebooks/`**: Analysis notebooks
  - `dataset_walkthrough.ipynb`: Main analysis notebook showing data exploration, neural response analysis, and clustering

### Key Data Loading Pattern

```python
import h5py
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc

# Load neural data
h5_file = h5py.File("GoodUnit_240709_JianJian_NSD1000_LOC_g2.mat", "r")
neural_data = load_data_from_GoodUnitStrc(h5_file)
```

### Neural Data Structure (from GoodUnit files)

Key arrays returned by `load_data_from_GoodUnitStrc`:
- `Raster`: Shape (n_units, n_timepoints, n_trials) - spike raster data
- `response_matrix_img`: Shape (n_units, n_timepoints, n_images) - averaged responses
- `unittype`: Shape (n_units,) - unit classifications
- `trial_valid_idx`: Valid trial indices for stimulus mapping
- `dataset_valid_idx`: Valid dataset indices

### Analysis Workflow

The typical analysis pattern involves:
1. Loading neural data from GoodUnit files
2. Extracting valid trials and mapping to stimulus indices  
3. Computing baseline and evoked responses
4. Building response matrices for analysis (clustering, similarity)
5. Loading corresponding image stimuli for visualization

## Common Development Tasks

### Loading and Exploring Data
```python
# Basic data loading
from agent_out.data_loader import NSDDataLoader
loader = NSDDataLoader("/path/to/nsd/data")
data = loader.load_file("GoodUnit_file.mat")

# For detailed exploration
from NSD_utils.h5_dataset_utils import explore_h5_structure
explore_h5_structure(dataset, h5_file, name='root')
```

### Working with Neural Responses
```python
# Extract time windows for analysis
evk_slice = slice(100, 250)  # evoked response window
bsl_slice = slice(0, 90)     # baseline window

# Compute firing rates (multiply by 1000 for spikes/sec)
evk_resp = Raster[:, evk_slice, :].mean(axis=1) * 1000
```

### Image Dataset Integration
```python
from torchvision.datasets import ImageFolder
nsd_dataset = ImageFolder(root=data_root, transform=None)
```

## File Organization

- Data files follow strict naming conventions documented in `agent_out/README.md`
- Images stored in `NSD1000_LOC/` subdirectory
- All analysis should use the utilities in `NSD_utils/` for consistent data loading
- The project has no formal package structure - use relative imports or add to Python path

## Testing and Validation

- No formal test suite - validation done through notebooks
- Main validation in `notebooks/dataset_walkthrough.ipynb`
- Check data shapes and statistics after loading to verify correctness