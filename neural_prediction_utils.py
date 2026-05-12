"""
Neural Prediction Utilities

This module provides utilities for predicting neural responses using deep networks,
migrated and adapted from the Closed-loop-visual-insilico repository.

Key functionality:
- Feature extraction from deep networks
- Dimensionality reduction (PCA, SRP, spatial pooling)
- Ridge/Lasso regression sweeps for neural prediction
- Model evaluation metrics

Author: Migrated from Closed-loop-visual-insilico
Date: 2025
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from copy import deepcopy
from collections import defaultdict
from typing import Dict, List, Any, Union, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import resnet50

from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, MultiTaskLassoCV
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import timm
except ImportError:
    timm = None
    print("Warning: timm not available. Some model loading functions may not work.")

try:
    from circuit_toolkit.layer_hook_utils import featureFetcher
    from circuit_toolkit.dataset_utils import ImagePathDataset
except ImportError:
    print("Warning: circuit_toolkit not available. Using fallback implementations.")
    featureFetcher = None
    ImagePathDataset = None


# ============================================================================
# Model Loading Utils
# ============================================================================

def load_model_transform(modelname: str, device: str = "cuda") -> Tuple[torch.nn.Module, Any]:
    """
    Load model and associated transforms for neural prediction.
    
    Args:
        modelname: Name of the model ('resnet50', 'resnet50_robust', etc.)
        device: Device to load model on
        
    Returns:
        model: PyTorch model
        transforms_pipeline: Transform pipeline for preprocessing
    """
    if modelname == "resnet50_robust":
        model = resnet50(pretrained=False)
        # Note: You'll need to provide the path to the robust model checkpoint
        # model.load_state_dict(torch.load("/path/to/imagenet_linf_8_pure.pt"))
        transforms_pipeline = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif modelname == "resnet50":
        model = resnet50(pretrained=True)
        transforms_pipeline = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif modelname == "dinov2_vitb14_reg":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        transforms_pipeline = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unknown model: {modelname}")
        
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, transforms_pipeline


# ============================================================================
# Feature Transform Functions
# ============================================================================

def avgtoken_transform(x):
    """Average pooling across token dimension for transformers."""
    return x.mean(dim=(1))

def clstoken_transform(x):
    """Extract CLS token for transformers."""
    return x[:, 0, :]

def flatten_transform(x):
    """Flatten spatial/token dimensions."""
    return x.flatten(start_dim=1)

def sp_avg_transform(x):
    """Spatial average pooling for CNNs."""
    return x.mean(dim=(2,3))

def sp_cent_transform(x):
    """Center spatial pooling for CNNs."""
    centpos = (x.shape[2] // 2, x.shape[3] // 2)
    return x[:, :, centpos[0]:centpos[0]+1, centpos[1]:centpos[1]+1].mean(dim=(2,3))


# ============================================================================
# Custom Regressor Classes
# ============================================================================

class MultiOutputSeparateLassoCV(BaseEstimator, RegressorMixin):
    """Separate LassoCV for each output unit."""
    
    def __init__(self, alphas=None, cv=5, max_iter=1000, random_state=None):
        self.alphas = alphas
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        self.n_outputs_ = y.shape[1]
        self.estimators_ = []
        self.alpha_ = []
        
        for i in range(self.n_outputs_):
            lasso = LassoCV(alphas=self.alphas, cv=self.cv, 
                           max_iter=self.max_iter, random_state=self.random_state)
            lasso.fit(X, y[:, i])
            self.estimators_.append(lasso)
            self.alpha_.append(lasso.alpha_)
            
        self.alpha_ = np.array(self.alpha_)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([est.predict(X) for est in self.estimators_])
        return predictions.squeeze() if predictions.shape[1] == 1 else predictions
    
    def score(self, X, y):
        y_pred = self.predict(X)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        return 1 - np.square(y - y_pred).sum() / np.square(y - y.mean(axis=0)).sum()


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def calc_reduce_features_dataset(dataset, feat_transformers, net, featlayer,
                                batch_size=40, workers=6, img_dim=(227, 227), idx_range=None):
    """
    Calculate reduced features for a dataset with various transformations.
    
    Args:
        dataset: Image Dataset
        feat_transformers: Dict of functions that reduce feature tensors
            Examples:
                {"pca": lambda tsr: pca.transform(tsr.reshape(tsr.shape[0], -1)),
                 "srp": lambda tsr: srp.transform(tsr.reshape(tsr.shape[0], -1)),
                 "sp_avg": lambda tsr: tsr.mean(axis=(2, 3))}
        net: Network to extract features from
        featlayer: Layer to extract features from
        batch_size: Batch size for DataLoader
        workers: Number of workers for DataLoader
        img_dim: Image dimensions
        idx_range: Range of indices to process (optional)
        
    Returns:
        Dictionary of transformed features for each transformer
    """
    if featureFetcher is None:
        raise ImportError("circuit_toolkit.layer_hook_utils.featureFetcher not available")
        
    if idx_range is None:
        imgloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    else:
        imgloader = DataLoader(Subset(dataset, idx_range), batch_size=batch_size,
                              shuffle=False, num_workers=workers)

    featFetcher = featureFetcher(net, print_module=False)
    featFetcher.record(featlayer)
    feattsr_col = defaultdict(list)
    
    for i, (imgtsr, score) in tqdm(enumerate(imgloader)):
        with torch.no_grad():
            net(imgtsr.cuda())
        feattsr = featFetcher[featlayer]
        for tfmname, feat_transform in feat_transformers.items():
            feattsr_col[tfmname].append(feat_transform(feattsr.cpu().numpy()))
    
    for tfmname in feattsr_col:
        feattsr_col[tfmname] = np.concatenate(feattsr_col[tfmname], axis=0)
        print(tfmname, "feature tensor shape", feattsr_col[tfmname].shape)
    
    del featFetcher
    return feattsr_col


def transform_features2Xdict(feat_dict, layer_names=None, 
                           dimred_list=["pca1000", "sp_cent", "sp_avg", "full"],
                           pretrained_Xtransforms={}, use_pca_dual=False,
                           train_split_idx=None):
    """
    Transform features using various dimensionality reduction methods.
    
    Args:
        feat_dict: Dictionary of features per layer
        layer_names: List of layer names to process (None for all)
        dimred_list: List of dimensionality reduction methods
        pretrained_Xtransforms: Dict of pretrained transformers
        use_pca_dual: Whether to use dual PCA solver (requires additional library)
        train_split_idx: Indices for training split
        
    Returns:
        Xdict: Dictionary of transformed features
        tfm_dict: Dictionary of fitted transformers
    """
    Xdict = {}
    tfm_dict = {}
    time_start = time.time()
    
    for layerkey in feat_dict.keys() if layer_names is None else layer_names:
        feat_tsr = feat_dict[layerkey]
        time_feat_tsr = time.time()
        print(layerkey, feat_tsr.shape)
        
        # Convert to numpy if torch tensor
        if isinstance(feat_tsr, torch.Tensor):
            feat_tsr = feat_tsr.cpu().numpy()
            
        featmat = feat_tsr.reshape(feat_tsr.shape[0], -1)  # Flatten spatial dims
        
        if train_split_idx is not None:
            featmat_train = featmat[train_split_idx, :]
        else:
            featmat_train = featmat
        
        for dimred in dimred_list:
            time_dimred = time.time()
            
            if dimred.startswith("pca"):
                n_components = int(dimred.split("pca")[-1])
                if f"{layerkey}_{dimred}" in pretrained_Xtransforms:
                    # Use pretrained PCA transformer
                    pca_transformer = pretrained_Xtransforms[f"{layerkey}_{dimred}"]
                    featmat_pca = pca_transformer.transform(featmat)
                else:
                    # Fit PCA transformer on training set
                    pca_transformer = PCA(n_components=n_components)
                    pca_transformer.fit(featmat_train)
                    featmat_pca = pca_transformer.transform(featmat)
                
                Xdict[f"{layerkey}_{dimred}"] = featmat_pca
                tfm_dict[f"{layerkey}_{dimred}"] = pca_transformer
                X_shape = featmat_pca.shape
                
            elif dimred.startswith("srp"):
                if dimred == "srp":
                    n_components = "auto"
                else:
                    n_components = int(dimred.split("srp")[-1])
                
                if f"{layerkey}_{dimred}" in pretrained_Xtransforms:
                    srp_transformer = pretrained_Xtransforms[f"{layerkey}_{dimred}"]
                    featmat_srp = srp_transformer.transform(featmat)
                else:
                    srp_transformer = SparseRandomProjection(n_components=n_components, random_state=42)
                    featmat_srp = srp_transformer.fit_transform(featmat)
                
                Xdict[f"{layerkey}_{dimred}"] = featmat_srp
                tfm_dict[f"{layerkey}_{dimred}"] = srp_transformer
                X_shape = featmat_srp.shape
                
            elif dimred == "sp_avg":
                if len(feat_tsr.shape) == 4:  # Conv features: B x C x H x W
                    featmat_avg = feat_tsr.mean(axis=(2, 3))
                elif len(feat_tsr.shape) == 3:  # Transformer features: B x T x C
                    featmat_avg = feat_tsr.mean(axis=1)
                else:
                    featmat_avg = featmat
                
                Xdict[f"{layerkey}_sp_avg"] = featmat_avg
                tfm_dict[f"{layerkey}_sp_avg"] = sp_avg_transform
                X_shape = featmat_avg.shape
                
            elif dimred == "sp_cent":
                if len(feat_tsr.shape) == 4:  # Conv features
                    centpos = (feat_tsr.shape[2] // 2, feat_tsr.shape[3] // 2)
                    featmat_cent = feat_tsr[:, :, centpos[0]:centpos[0]+1, centpos[1]:centpos[1]+1].mean(axis=(2,3))
                else:
                    featmat_cent = featmat  # Fallback to flattened
                
                Xdict[f"{layerkey}_sp_cent"] = featmat_cent
                tfm_dict[f"{layerkey}_sp_cent"] = sp_cent_transform
                X_shape = featmat_cent.shape
                
            elif dimred == "clstoken":
                if len(feat_tsr.shape) == 3:  # Transformer features: B x T x C
                    featmat_cls = feat_tsr[:, 0, :]  # CLS token
                else:
                    featmat_cls = featmat  # Fallback
                
                Xdict[f"{layerkey}_clstoken"] = featmat_cls
                tfm_dict[f"{layerkey}_clstoken"] = clstoken_transform
                X_shape = featmat_cls.shape
                
            elif dimred == "avgtoken":
                if len(feat_tsr.shape) == 3:  # Transformer features: B x T x C
                    featmat_avg = feat_tsr.mean(axis=1)
                else:
                    featmat_avg = featmat
                
                Xdict[f"{layerkey}_avgtoken"] = featmat_avg
                tfm_dict[f"{layerkey}_avgtoken"] = avgtoken_transform
                X_shape = featmat_avg.shape
                
            elif dimred == "full":
                Xdict[f"{layerkey}_full"] = featmat
                tfm_dict[f"{layerkey}_full"] = flatten_transform
                X_shape = featmat.shape
                
            else:
                raise ValueError(f"Unknown dimension reduction method: {dimred}")
                
            print(f"Time taken to transform {layerkey} {dimred} {list(X_shape)}: {time.time() - time_dimred:.3f}s")
        
        print(f"Time taken to transform {layerkey}: {time.time() - time_feat_tsr:.3f}s")
    
    print(f"Time taken to transform all features: {time.time() - time_start:.3f}s")
    return Xdict, tfm_dict


# ============================================================================
# Regression Functions
# ============================================================================

def sweep_regressors(Xdict, y_all, regressors, regressor_names, 
                    verbose=True, n_jobs=-1, train_split_idx=None):
    """
    Sweep through regressors and feature types with cross-validation.
    
    Args:
        Xdict: Dictionary of feature matrices
        y_all: Target neural responses
        regressors: List of regressor objects
        regressor_names: List of regressor names
        verbose: Whether to print progress
        n_jobs: Number of parallel jobs
        train_split_idx: Training indices (None for random split)
        
    Returns:
        result_df: DataFrame with regression results
        models: Dictionary of fitted models
    """
    result_summary = {}
    models = {}
    
    if train_split_idx is None:
        idx_train, idx_test = train_test_split(
            np.arange(len(y_all)), test_size=0.2, random_state=42, shuffle=True
        )
    else:
        idx_train = train_split_idx
        idx_test = np.setdiff1d(np.arange(len(y_all)), train_split_idx)
        print(f"Using {len(idx_train)} training samples and {len(idx_test)} testing samples")
    
    for xtype in Xdict:
        X_all = Xdict[xtype]
        y_train, y_test = y_all[idx_train], y_all[idx_test]
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        nfeat = X_train.shape[1]
        
        for estim, label in zip(regressors, regressor_names):
            start_time = time.time()
            
            if isinstance(estim, (RidgeCV, LassoCV, MultiTaskLassoCV)):
                clf = estim.fit(X_train, y_train)
                clf = deepcopy(clf)
                alpha = estim.alpha_
            elif isinstance(estim, MultiOutputSeparateLassoCV):
                clf = estim.fit(X_train, y_train)
                clf = deepcopy(clf)
                alpha = estim.alpha_
            elif hasattr(estim, "alpha"):
                clf = GridSearchCV(
                    estimator=estim, n_jobs=n_jobs,
                    param_grid=dict(alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5, 1E6, 1E7, 1E8, 1E9])
                ).fit(X_train, y_train)
                alpha = clf.best_params_["alpha"]
            else:
                clf = estim.fit(X_train, y_train)
                clf = deepcopy(clf)
                alpha = np.nan
            
            D2_train = clf.score(X_train, y_train)
            D2_test = clf.score(X_test, y_test)
            end_time = time.time()
            
            result_summary[(xtype, label)] = {
                "alpha": alpha, 
                "train_score": D2_train, 
                "test_score": D2_test, 
                "n_feat": nfeat, 
                "runtime": end_time - start_time
            }
            models[(xtype, label)] = clf
            
            if verbose:
                print(f"{xtype} {label} D2_train: {D2_train:.3f} D2_test: {D2_test:.3f} time: {end_time - start_time:.3f}")

    result_df = pd.DataFrame(result_summary)
    print(result_df.T)
    return result_df.T, models


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_R2_per_unit(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute R² (coefficient of determination) per unit/neuron.
    
    Args:
        y_true: True responses, shape (n_samples, n_units)
        y_pred: Predicted responses, shape (n_samples, n_units)
        
    Returns:
        R² values per unit, shape (n_units,)
    """
    return 1 - np.square(y_true - y_pred).sum(axis=0) / np.square(y_true - y_true.mean(axis=0)).sum(axis=0)


def evaluate_prediction(fit_models, Xfeat_dict, y_true, label="", savedir=None):
    """
    Evaluate the prediction of a dict of models.
    
    Args:
        fit_models: Dictionary of fitted models
        Xfeat_dict: Dictionary of feature tensors
        y_true: True y values
        label: Label for saving
        savedir: Directory to save results
        
    Returns:
        df: DataFrame summarizing evaluation statistics
        eval_dict: Same statistics in dict format
        y_pred_dict: Dictionary of prediction vectors
    """
    print(label, f"  N imgs: {len(y_true)}")
    eval_dict = {}
    y_pred_dict = {}
    
    for (Xtype, regrname), regr in fit_models.items():
        try:
            y_pred = regr.predict(Xfeat_dict[Xtype])
            D2 = regr.score(Xfeat_dict[Xtype], y_true)
            rho_p, pval_p = pearsonr(y_pred.flatten(), y_true.flatten())
            rho_s, pval_s = spearmanr(y_pred.flatten(), y_true.flatten())
            
            print(f"{Xtype} {regrname} Prediction Pearson: {rho_p:.3f} {pval_p:.1e} Spearman: {rho_s:.3f} {pval_s:.1e} D2: {D2:.3f}")
            
            eval_dict[(Xtype, regrname)] = {
                "rho_p": rho_p, "pval_p": pval_p, "rho_s": rho_s, "pval_s": pval_s, 
                "D2": D2, "imgN": len(y_true)
            }
            y_pred_dict[(Xtype, regrname)] = y_pred
        except Exception as e:
            print(f"Error evaluating {Xtype} {regrname}: {e}")
            continue
    
    # Parse label for additional metadata
    parts = label.split("-")
    layer = parts[-2] if len(parts) >= 2 else ""
    datasetstr = parts[-1] if len(parts) >= 1 else ""
    
    df = pd.DataFrame(eval_dict).T
    df["label"] = label
    df["layer"] = layer
    df["img_space"] = datasetstr
    
    if savedir is not None:
        df.to_csv(os.path.join(savedir, f"eval_predict_{label}.csv"), index=True)
        
    return df, eval_dict, y_pred_dict


def compare_activation_prediction(target_scores, pred_scores_dict, exptitle="", savedir=""):
    """
    Compare target scores with predicted scores across different methods.
    
    Args:
        target_scores: True target scores
        pred_scores_dict: Dictionary of predicted scores per method
        exptitle: Experiment title
        savedir: Directory to save results
        
    Returns:
        DataFrame with comparison results
    """
    result_col = {}
    
    for k in pred_scores_dict:
        rho_s = spearmanr(target_scores, pred_scores_dict[k])
        rho_p = pearsonr(target_scores, pred_scores_dict[k])
        R2 = 1 - np.var(pred_scores_dict[k] - target_scores) / np.var(target_scores)
        
        print(k, f"spearman: {rho_s.correlation:.3f} P={rho_s.pvalue:.1e}",
                 f"pearson: {rho_p[0]:.3f} P={rho_p[1]:.1e} R2={R2:.3f}")
        
        result_col[k] = {
            "spearman": rho_s.correlation, "pearson": rho_p[0],
            "spearman_pval": rho_s.pvalue, "pearson_pval": rho_p[1],
            "R2": R2, "dataset": exptitle, "n_sample": len(target_scores)
        }

        plt.figure(figsize=(6, 6))
        plt.scatter(target_scores, pred_scores_dict[k], s=16, alpha=0.5)
        plt.xlabel("Target scores")
        plt.ylabel("Predicted scores")
        plt.axis('equal')
        plt.title(f"{exptitle} {k}\n"
                  f"corr pearsonr {rho_p[0]:.3f} P={rho_p[1]:.1e}\n"
                  f"corr spearmanr {rho_s.correlation:.3f} P={rho_s.pvalue:.1e} R2={R2:.3f}")
        plt.tight_layout()
        
        if savedir:
            plt.savefig(os.path.join(savedir, f"{exptitle}_{k}_regress.png"))
        plt.show()

    test_result_df = pd.DataFrame(result_col)
    if savedir:
        test_result_df.T.to_csv(os.path.join(savedir, f"{exptitle}_regress_results.csv"))
    
    return test_result_df.T


# ============================================================================
# Complete Pipeline Functions
# ============================================================================

def neural_prediction_pipeline(features: np.ndarray, responses: np.ndarray,
                              dimred_methods: List[str] = ['pca1000', 'sp_avg'],
                              regression_methods: List[str] = ['RidgeCV', 'LassoCV'],
                              test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Complete pipeline for neural prediction with multiple methods.
    
    Args:
        features: Input features, shape (n_samples, n_features) or dict of features
        responses: Neural responses, shape (n_samples, n_units)
        dimred_methods: List of dimensionality reduction methods
        regression_methods: List of regression methods
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary containing results from all method combinations
    """
    # Split data
    indices = np.arange(len(responses))
    idx_train, idx_test = train_test_split(indices, test_size=test_size, 
                                          random_state=random_state, shuffle=True)
    
    # If features is an array, convert to dict
    if isinstance(features, np.ndarray):
        feat_dict = {'layer1': features}
    else:
        feat_dict = features
    
    # Transform features
    Xdict, tfm_dict = transform_features2Xdict(
        feat_dict, 
        dimred_list=dimred_methods,
        train_split_idx=idx_train
    )
    
    # Prepare regressors
    alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]
    regressors = []
    regressor_names = []
    
    for method in regression_methods:
        if method == 'RidgeCV':
            # Use alpha_per_target without specifying cv (uses default cv=None for alpha_per_target)
            regressors.append(RidgeCV(alphas=alpha_list, alpha_per_target=True))
            regressor_names.append('RidgeCV')
        elif method == 'LassoCV':
            regressors.append(MultiTaskLassoCV(alphas=alpha_list, cv=5))
            regressor_names.append('LassoCV')
        elif method == 'SeparateLassoCV':
            regressors.append(MultiOutputSeparateLassoCV(alphas=alpha_list, cv=5))
            regressor_names.append('SeparateLassoCV')
    
    # Sweep through regressors
    result_df, models = sweep_regressors(
        Xdict, responses, regressors, regressor_names,
        train_split_idx=idx_train
    )
    
    # Evaluate models
    eval_df, eval_dict, y_pred_dict = evaluate_prediction(
        models, Xdict, responses, label="neural_prediction"
    )
    
    return {
        'result_df': result_df,
        'eval_df': eval_df,
        'models': models,
        'transformers': tfm_dict,
        'Xdict': Xdict,
        'predictions': y_pred_dict,
        'splits': {'idx_train': idx_train, 'idx_test': idx_test}
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_prediction_results(y_true: np.ndarray, y_pred: np.ndarray, 
                           title: str = "Neural Prediction Results",
                           unit_idx: Optional[int] = None, save_path: Optional[str] = None):
    """
    Plot neural prediction results.
    
    Args:
        y_true: True responses
        y_pred: Predicted responses  
        title: Plot title
        unit_idx: Specific unit to plot (if None, plot average across units)
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    if y_true.ndim > 1 and unit_idx is not None:
        # Plot specific unit
        plt.scatter(y_true[:, unit_idx], y_pred[:, unit_idx], alpha=0.6, s=20)
        plt.xlabel(f"True Response (Unit {unit_idx})")
        plt.ylabel(f"Predicted Response (Unit {unit_idx})")
        
        # Compute correlation for this unit
        corr = pearsonr(y_true[:, unit_idx], y_pred[:, unit_idx])[0]
        r2 = compute_R2_per_unit(y_true[:, unit_idx:unit_idx+1], y_pred[:, unit_idx:unit_idx+1])[0]
        plt.title(f"{title}\nUnit {unit_idx}: r={corr:.3f}, R²={r2:.3f}")
        
    elif y_true.ndim > 1:
        # Plot average across units
        y_true_mean = np.mean(y_true, axis=1)
        y_pred_mean = np.mean(y_pred, axis=1)
        plt.scatter(y_true_mean, y_pred_mean, alpha=0.6, s=20)
        plt.xlabel("True Response (Mean)")
        plt.ylabel("Predicted Response (Mean)")
        
        corr = pearsonr(y_true_mean, y_pred_mean)[0]
        plt.title(f"{title}\nMean across units: r={corr:.3f}")
        
    else:
        # Single unit case
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        plt.xlabel("True Response")
        plt.ylabel("Predicted Response")
        
        corr = pearsonr(y_true, y_pred)[0]
        r2 = 1 - np.var(y_pred - y_true) / np.var(y_true)
        plt.title(f"{title}\nr={corr:.3f}, R²={r2:.3f}")
    
    # Add diagonal line
    min_val = min(plt.xlim()[0], plt.ylim()[0])
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Neural Prediction Utils loaded successfully!")
    print("\nKey functions available:")
    print("- neural_prediction_pipeline(): Complete pipeline for neural prediction")
    print("- sweep_regressors(): Sweep through multiple regressors and feature types")
    print("- transform_features2Xdict(): Transform features with various reduction methods")
    print("- evaluate_prediction(): Comprehensive model evaluation")
    print("- compare_activation_prediction(): Compare predictions across methods")
    print("- plot_prediction_results(): Visualization utilities")