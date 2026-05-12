import h5py
import numpy as np
# Parse and explore the structure
def explore_h5_structure(dataset, group, name='', level=0, ):
    indent = "  " * level
    if isinstance(group, h5py.Group):
        print(f"{indent}{name}/ (Group)")
        for key in group.keys():
            if key not in ['#refs#', '#subsystem#']:
                explore_h5_structure(dataset, group[key], key, level + 1)
    elif isinstance(group, h5py.Dataset):
        print(f"{indent}{name}: shape {group.shape}, dtype {group.dtype}")
        if group.dtype == np.dtype('O'):
            # print(f"{indent}{name}: first element")
            explore_h5_structure(dataset, group[0,0], '(0,0)', level + 1)
    elif isinstance(group, h5py.Reference):
        print(f"{indent}{name}: Reference")
        explore_h5_structure(dataset, dataset[group], "value", level)
        
        
def concat_ref_dataset(h5_file, ref_array):
    ref_array_shape = ref_array.shape
    assert ref_array.dtype == np.dtype('O')
    concat_data = []
    for i in range(ref_array_shape[0]):
        for j in range(ref_array_shape[1]):
            ref = ref_array[i,j]
            ref_data = h5_file[ref][:]
            concat_data.append(ref_data)
    concat_data = np.stack(concat_data, axis=0)
    entry_shape = concat_data.shape[1:]
    return concat_data.reshape(*ref_array_shape, *entry_shape)



def load_data_from_GoodUnitStrc(h5_file):
    exp_subject = "".join([chr(x) for x in h5_file['meta_data']["exp_subject"][:].flatten()])
    trial_valid_idx = h5_file['meta_data']["trial_valid_idx"][:].astype(int) # (n_trials_all, )
    print("trial_valid_idx", trial_valid_idx.shape)
    dataset_valid_idx = h5_file['meta_data']["dataset_valid_idx"][:].astype(bool) # (n_trials_all, )
    print("dataset_valid_idx", dataset_valid_idx.shape)
    Raster = concat_ref_dataset(h5_file, h5_file['GoodUnitStrc']['Raster']).squeeze() # (n_units, n_timepoints, n_trials)
    print("Raster", Raster.shape)
    response_matrix_img = concat_ref_dataset(h5_file, h5_file['GoodUnitStrc']['response_matrix_img']).squeeze() # (n_units, n_timepoints, n_images[average])
    print("response_matrix_img", response_matrix_img.shape)
    unittype = concat_ref_dataset(h5_file, h5_file['GoodUnitStrc']['unittype']).squeeze() # (n_units, )
    print("unittype", unittype.shape)
    KSidx = concat_ref_dataset(h5_file, h5_file['GoodUnitStrc']['KSidx']).squeeze() # (n_units, )
    print("KSidx", KSidx.shape)
    spikepos = concat_ref_dataset(h5_file, h5_file['GoodUnitStrc']['spikepos']).squeeze() # (n_units, )
    print("spikepos", spikepos.shape)
    PsthRange = h5_file["global_params"]["PsthRange"][:].squeeze()
    print("PsthRange", PsthRange.shape)
    # check the shape matches
    n_units = Raster.shape[0]
    n_timepoints = Raster.shape[1]
    n_valid_trials = Raster.shape[2]
    n_images = response_matrix_img.shape[2]
    n_trials = trial_valid_idx.shape[0]
    assert (dataset_valid_idx.sum() == n_valid_trials)
    assert (len(np.unique(trial_valid_idx)) == n_images+1)
    assert (trial_valid_idx[~dataset_valid_idx] == 0).all()
    assert (len(np.unique(trial_valid_idx[dataset_valid_idx])) == n_images)
    assert len(unittype) == n_units
    assert len(KSidx) == n_units
    assert len(spikepos) == n_units
    assert len(PsthRange) == n_timepoints
    structure = {
        "exp_subject": exp_subject,
        "trial_valid_idx": trial_valid_idx,
        "dataset_valid_idx": dataset_valid_idx,
        "Raster": Raster,
        "response_matrix_img": response_matrix_img,
        "PsthRange": PsthRange,
        "unittype": unittype,
        "KSidx": KSidx,
        "spikepos": spikepos,
        "n_units": n_units,
        "n_timepoints": n_timepoints,
        "n_valid_trials": n_valid_trials,
        "n_trials": n_trials,
        "n_images": n_images,
        }
    return structure

