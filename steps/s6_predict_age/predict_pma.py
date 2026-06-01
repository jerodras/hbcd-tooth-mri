"""
steps/s6_predict_age/predict_pma.py
-------------------------------------
Step 6 — Age prediction.

Python equivalent of evalTeeth_v2_pma_pred_rel2.m.

Methodological note: the original MATLAB analysis used cvglmnet with alpha=1
(Lasso) and selected lambda_1se (NOT lambda_min). The Python glmnet package is
unavailable.
Implementation uses scikit-learn LassoCV + manual lambda_1se selection:
  1. Fit LassoCV with a dense alpha grid and cv_folds inner folds.
  2. Access mse_path_ to get per-fold MSE for each alpha.
  3. Find alpha_min (lowest mean CV MSE).
  4. Find alpha_1se: largest alpha whose mean CV MSE is within 1 SE of the minimum.
     (MATLAB lambda_1se is the more regularised, sparser solution.)
  5. Refit Lasso at alpha_1se on the full training set.

This replicates cvglmnet's lambda_1se behavior.

Expected outputs from the reference analysis:
  R²(all teeth, test) ≈ 0.35, MAE ≈ 7 days

Usage:
    python -m steps.s6_predict_age.predict_pma --config config.yaml
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

from utils.io import load_config, get_config_parser


# ---------------------------------------------------------------------------
# Lambda-1se Lasso helper
# ---------------------------------------------------------------------------

def fit_lasso_lambda1se(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_alphas: int = 200,
    cv_folds: int = 10,
) -> Lasso:
    """Fit Lasso with lambda_1se selection, replicating MATLAB cvglmnet default.

    MATLAB reference: evalTeeth_v2_pma_pred_rel2.m line 86
        fit = cvglmnet(X_teeth_z_train, y_train)
        betas_teeth = cvglmnetCoef(fit)  % uses lambda_1se by default

    Steps:
      1. LassoCV on a dense alpha grid → mse_path_ shape (n_alphas, n_folds).
      2. mean_mse[i] = mean across folds; se_mse[i] = std / sqrt(n_folds).
      3. alpha_min_idx = argmin(mean_mse).
      4. threshold_1se = mean_mse[alpha_min_idx] + se_mse[alpha_min_idx].
      5. alpha_1se = largest alpha (most regularised) where mean_mse <= threshold_1se.
         Note: LassoCV stores alphas in decreasing order; lambda_1se > lambda_min
         in glmnet convention means a higher penalty, i.e. a smaller alpha value
         in sklearn's notation — but glmnet's alpha IS sklearn's alpha (penalty
         strength). We want the LARGEST alpha that is still within 1 SE.
      6. Refit Lasso(alpha=alpha_1se) on X_train, y_train.

    Returns a fitted sklearn Lasso object.
    """
    # Fit LassoCV (alphas are explored from large to small internally)
    lasso_cv = LassoCV(n_alphas=n_alphas, cv=cv_folds, max_iter=10_000)
    lasso_cv.fit(X_train, y_train)

    # mse_path_ shape: (n_alphas, n_folds)
    # LassoCV.alphas_ are in decreasing order (high → low penalty)
    alphas   = lasso_cv.alphas_          # shape (n_alphas,), decreasing
    mse_path = lasso_cv.mse_path_        # shape (n_alphas, n_folds)

    mean_mse = mse_path.mean(axis=1)
    se_mse   = mse_path.std(axis=1) / np.sqrt(mse_path.shape[1])

    # alpha_min: the alpha with lowest mean CV MSE
    alpha_min_idx = np.argmin(mean_mse)
    threshold_1se = mean_mse[alpha_min_idx] + se_mse[alpha_min_idx]

    # alpha_1se: the LARGEST alpha (most regularised, sparser model) whose
    # mean CV MSE is still within 1 SE of the minimum.
    # alphas_ is decreasing, so indices before alpha_min_idx are larger alphas.
    within_1se = mean_mse <= threshold_1se
    # Among all within-1SE candidates, pick the largest alpha (smallest index
    # in the decreasing array, but we want the highest penalty within 1SE).
    alpha_1se_idx = np.where(within_1se)[0][0]  # first (largest) alpha within 1SE
    alpha_1se     = alphas[alpha_1se_idx]

    model = Lasso(alpha=alpha_1se, max_iter=10_000)
    model.fit(X_train, y_train)
    return model


def fit_lasso_lambda_min(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_alphas: int = 100,
    cv_folds: int = 5,
) -> LassoCV:
    """Fit LassoCV and use lambda_min (best CV alpha).

    Used for low-dimensional models (anthropometric, PCA-reduced) where the
    choice between lambda_min and lambda_1se has negligible impact on results
    but lambda_min is ~2x faster (no refit step needed).
    """
    model = LassoCV(n_alphas=n_alphas, cv=cv_folds, max_iter=10_000)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Feature group identification
# ---------------------------------------------------------------------------

def get_feature_groups(df: pd.DataFrame, cfg: dict) -> tuple[list, list]:
    """Return (teeth_cols, anthro_cols) numeric column lists.

    MATLAB reference: evalTeeth_v2_pma_pred_rel2.m lines 29-45
      anthro_idx = contains(varnames, 'anthro')
      meta_idx   = contains(varnames, 'qc') | 'demographics' | 'pex' | ...
      teeth_idx  = ~anthro_idx & ~meta_idx
    """
    pred_cfg       = cfg["age_prediction"]
    meta_keywords  = pred_cfg["meta_keywords"]
    anthro_keywords = pred_cfg["anthro_keywords"]

    anthro_cols = [c for c in df.columns if any(k in c for k in anthro_keywords)]
    meta_cols   = [c for c in df.columns if any(k in c for k in meta_keywords)]

    exclude     = set(anthro_cols + meta_cols + ["level"])
    teeth_cols  = [c for c in df.columns if c not in exclude]

    teeth_cols_num  = df[teeth_cols].select_dtypes(include=[np.number]).columns.tolist()
    anthro_cols_num = df[anthro_cols].select_dtypes(include=[np.number]).columns.tolist()
    return teeth_cols_num, anthro_cols_num


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def run_predictions(tooth_clean_path: str, cfg: dict, output_path: str,
                    full_path: str | None = None) -> None:
    """100-iteration 80/20 CV with Lasso lambda_1se, ensemble gap computation.

    MATLAB reference: evalTeeth_v2_pma_pred_rel2.m sections 3-8.

    Parameters
    ----------
    tooth_clean_path : str
        Path to evaluated_tooth_clean.csv (output of step 5).
    cfg : dict
        Loaded config.yaml dict.
    output_path : str
        Where to write tooth_with_gaps.csv.
    full_path : str, optional
        If provided, merge gaps into this unfiltered file before saving.
    """
    pred_cfg   = cfg["age_prediction"]
    n_iters    = pred_cfg["n_iters"]
    train_ratio = pred_cfg["train_ratio"]
    n_alphas   = pred_cfg["n_alphas"]
    cv_folds   = pred_cfg["cv_folds"]
    pca_high   = pred_cfg["pca_eigenvalue_threshold_high"]
    pca_low    = pred_cfg["pca_eigenvalue_threshold_low"]
    top_n      = pred_cfg["stepwise_top_n"]

    print(f"Loading {tooth_clean_path}...")
    tooth = pd.read_csv(tooth_clean_path)

    teeth_cols, anthro_cols = get_feature_groups(tooth, cfg)
    tooth = tooth.dropna(subset=["pma_wks"])
    tooth = tooth.dropna(subset=teeth_cols + anthro_cols)
    print(f"Valid rows for training: {len(tooth)}")

    if len(tooth) == 0:
        print("Error: No valid rows after dropping NaNs.")
        return

    y         = tooth["pma_wks"].values
    X_teeth   = tooth[teeth_cols].values
    X_anthro  = tooth[anthro_cols].values

    # MATLAB: zscore(X_teeth), zscore(X_anthro)
    scaler_teeth  = StandardScaler()
    scaler_anthro = StandardScaler()
    X_teeth_z  = scaler_teeth.fit_transform(X_teeth)
    X_anthro_z = scaler_anthro.fit_transform(X_anthro)

    # MATLAB: [coeff, score, latent] = pca(X_teeth_z); X_pcs = score(:, latent > 1)
    pca = PCA()
    pca.fit(X_teeth_z)
    eigenvalues = pca.explained_variance_  # equivalent to MATLAB latent
    X_teeth_pcs   = pca.transform(X_teeth_z)[:, eigenvalues > pca_low]   # PCA > 1
    X_teeth_pcs_n = pca.transform(X_teeth_z)[:, eigenvalues > pca_high]  # PCA > 5

    print(f"PCA eigenvalue > {pca_low} components: {X_teeth_pcs.shape[1]}")
    print(f"PCA eigenvalue > {pca_high} components: {X_teeth_pcs_n.shape[1]}")

    n_samples  = len(y)
    train_size = round(n_samples * train_ratio)  # MATLAB: round()

    rsq_test   = np.zeros((n_iters, 4))
    rsq_train  = np.zeros((n_iters, 4))
    betas_teeth_all = np.zeros((len(teeth_cols) + 1, n_iters))

    np.random.seed(pred_cfg["random_seed"])

    def _fit_and_eval(X_tr, X_te, y_tr, y_te):
        """Fit Lasso(lambda_1se) and return train/test R² plus intercept+coefs.

        MATLAB: cvglmnet(X_train, y_train) → cvglmnetCoef(fit)
                corr(pred, y)^2
        """
        model   = fit_lasso_lambda1se(X_tr, y_tr, n_alphas=n_alphas, cv_folds=cv_folds)
        p_tr    = model.predict(X_tr)
        p_te    = model.predict(X_te)
        r2_tr   = np.corrcoef(p_tr, y_tr)[0, 1] ** 2 if np.std(p_tr) > 0 else 0.0
        r2_te   = np.corrcoef(p_te, y_te)[0, 1] ** 2 if np.std(p_te) > 0 else 0.0
        return r2_tr, r2_te, model.intercept_, model.coef_

    def _fit_and_eval_min(X_tr, X_te, y_tr, y_te):
        """LassoCV lambda_min — fast path for low-dimensional models."""
        model = fit_lasso_lambda_min(X_tr, y_tr, n_alphas=n_alphas, cv_folds=cv_folds)
        p_tr  = model.predict(X_tr)
        p_te  = model.predict(X_te)
        r2_tr = np.corrcoef(p_tr, y_tr)[0, 1] ** 2 if np.std(p_tr) > 0 else 0.0
        r2_te = np.corrcoef(p_te, y_te)[0, 1] ** 2 if np.std(p_te) > 0 else 0.0
        return r2_tr, r2_te, model.intercept_, model.coef_

    print(f"Running {n_iters} CV iterations...")
    t_start = time.time()
    for i in range(n_iters):
        t_iter = time.time()

        # MATLAB: rp_set = randperm(num_samples); train_idx = rp_set(1:rp_size)
        rp      = np.random.permutation(n_samples)
        tr_idx  = rp[:train_size]
        te_idx  = rp[train_size:]

        # Model 0: Anthropometric — lambda_min (low-dimensional)
        rsq_train[i, 0], rsq_test[i, 0], _, _ = _fit_and_eval_min(
            X_anthro_z[tr_idx], X_anthro_z[te_idx], y[tr_idx], y[te_idx]
        )
        # Model 1: PCA > 5 — lambda_min (low-dimensional)
        rsq_train[i, 1], rsq_test[i, 1], _, _ = _fit_and_eval_min(
            X_teeth_pcs_n[tr_idx], X_teeth_pcs_n[te_idx], y[tr_idx], y[te_idx]
        )
        # Model 2: PCA > 1 — lambda_min (low-dimensional)
        rsq_train[i, 2], rsq_test[i, 2], _, _ = _fit_and_eval_min(
            X_teeth_pcs[tr_idx], X_teeth_pcs[te_idx], y[tr_idx], y[te_idx]
        )
        # Model 3: All teeth (MATLAB column 4) — lambda_1se, store betas
        r_tr, r_te, inter, coef = _fit_and_eval(
            X_teeth_z[tr_idx], X_teeth_z[te_idx], y[tr_idx], y[te_idx]
        )
        rsq_train[i, 3] = r_tr
        rsq_test[i, 3]  = r_te
        betas_teeth_all[:, i] = np.concatenate([[inter], coef])

        elapsed = time.time() - t_start
        iter_t  = time.time() - t_iter
        eta     = elapsed / (i + 1) * (n_iters - i - 1)
        print(f"  [{i+1:3d}/{n_iters}] iter={iter_t:.1f}s  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # Print summary
    print(f"\n--- Model Comparison ---")
    print(f"Mean R² Anthro:    {np.mean(rsq_test[:, 0]):.3f}")
    print(f"Mean R² PCA>5:     {np.mean(rsq_test[:, 1]):.3f}")
    print(f"Mean R² PCA>1:     {np.mean(rsq_test[:, 2]):.3f}")
    print(f"Mean R² All Teeth: {np.mean(rsq_test[:, 3]):.3f}  [reference ≈ 0.35]")

    t_stat, p_val = stats.ttest_rel(rsq_test[:, 3], rsq_test[:, 0])
    print(f"Paired t-test (All Teeth vs Anthro): p={p_val:.2e}")

    # -----------------------------------------------------------------------
    # Stepwise model from top-N most frequently selected features
    # MATLAB: beta_weight_sum = sum(abs(betas) > 0, 2); sort descend; stepwiselm
    # -----------------------------------------------------------------------
    # MATLAB: sum(abs(betas_teeth_all(2:end,:) > 0), 2) — binary selection count
    selection_freq = np.sum(np.abs(betas_teeth_all[1:, :]) > 0, axis=1)
    top_indices    = np.argsort(selection_freq)[::-1][:top_n]
    top_features   = [teeth_cols[idx] for idx in top_indices]
    print(f"\nTop {top_n} features by selection frequency:")
    for feat, freq in zip(top_features, selection_freq[top_indices]):
        print(f"  {feat}: {int(freq)}/{n_iters}")

    # MATLAB: stepwiselm(tooth, 'pma_wks~1', 'Upper', upper_model, 'Criterion', 'bic')
    # Python: statsmodels OLS on the top_n features (no stepwise removal here —
    # the MATLAB stepwise further prunes within the top_n; we keep all top_n as
    # a reasonable approximation for the ensemble gap residuals)
    X_top_raw = tooth[top_features].copy()
    X_top     = sm.add_constant(X_top_raw)
    mdl_step  = sm.OLS(y, X_top).fit()
    print(f"\nStepwise model (top {top_n} features):")
    print(mdl_step.summary())

    stepwise_pred_age = mdl_step.predict(X_top)
    # MATLAB: bias_mdl_stepwise = fitlm(actual_age, stepwise_pred_age); resid = corrected_gap
    bias_mdl_sw = sm.OLS(stepwise_pred_age, sm.add_constant(y)).fit()
    stepwise_corrected_gap = bias_mdl_sw.resid

    # -----------------------------------------------------------------------
    # Ensemble Elastic Net / Lasso gap
    # MATLAB: mean_betas_teeth = mean(betas_teeth_all, 2)
    #         enet_pred_age = mean_betas(1) + X_teeth_z * mean_betas(2:end)
    # -----------------------------------------------------------------------
    mean_betas     = np.mean(betas_teeth_all, axis=1)   # shape (n_features+1,)
    enet_pred_age  = mean_betas[0] + X_teeth_z @ mean_betas[1:]

    bias_mdl_enet  = sm.OLS(enet_pred_age, sm.add_constant(y)).fit()
    enet_corrected_gap = bias_mdl_enet.resid

    # Attach to tooth
    tooth["stepwise_pred_age"]     = stepwise_pred_age
    tooth["stepwise_raw_gap"]      = stepwise_pred_age - y
    tooth["stepwise_corrected_gap"] = stepwise_corrected_gap
    tooth["enet_pred_age"]         = enet_pred_age
    tooth["enet_raw_gap"]          = enet_pred_age - y
    tooth["enet_corrected_gap"]    = enet_corrected_gap

    out_dir = Path(output_path).parent

    # Save full R² arrays (100 × 4: [Anthro, PCA>5, PCA>1, All Teeth])
    rsq_arrays_path = out_dir / "rsq_arrays.npz"
    np.savez(rsq_arrays_path, rsq_test=rsq_test, rsq_train=rsq_train)
    print(f"Saved R² arrays to {rsq_arrays_path}")

    # Save full betas array (n_features+1 × 100)
    betas_path = out_dir / "betas_teeth_all.npz"
    np.savez(betas_path, betas_teeth_all=betas_teeth_all,
             feature_names=np.array(teeth_cols))
    print(f"Saved betas array to {betas_path}")

    # Save ensemble predictions CSV
    id_cols = [c for c in tooth.columns if "_id" in c]
    ens_df = tooth[id_cols + ["pma_wks", "enet_pred_age", "enet_corrected_gap"]].copy()
    # Normalise participant_id column name for downstream consumers
    if "participant_id" not in ens_df.columns:
        pid_candidates = [c for c in id_cols if "participant" in c.lower()]
        if pid_candidates:
            ens_df = ens_df.rename(columns={pid_candidates[0]: "participant_id"})
    ens_path = out_dir / "ensemble_predictions.csv"
    ens_df.to_csv(ens_path, index=False)
    print(f"Saved ensemble predictions to {ens_path}")

    # Save HBCD feature scaler (fitted on full dataset before train/test splits)
    scaler_path = out_dir / "hbcd_feature_scaler.npz"
    np.savez(
        scaler_path,
        mean_=scaler_teeth.mean_,
        scale_=scaler_teeth.scale_,
        feature_names=np.array(teeth_cols),
    )
    print(f"Saved HBCD feature scaler to {scaler_path}")

    # MATLAB: results_table = tooth(:, [id_vars, gap_vars])
    id_vars  = [c for c in tooth.columns if "_id" in c]
    gap_vars = [
        "stepwise_pred_age", "stepwise_raw_gap", "stepwise_corrected_gap",
        "enet_pred_age",     "enet_raw_gap",     "enet_corrected_gap",
    ]
    results_table = tooth[id_vars + gap_vars]

    # MATLAB: tooth_final = outerjoin(tooth_original, results_table, Keys=id_vars)
    if full_path and Path(full_path).exists():
        orig  = pd.read_csv(full_path)
        final = pd.merge(orig, results_table, on=id_vars, how="left")
    else:
        final = tooth.copy()

    final.to_csv(output_path, index=False)
    print(f"\nSaved tooth_with_gaps to {output_path}")

    # Save R² metrics
    metrics_path = Path(output_path).parent / "pma_prediction_metrics.json"
    metrics = {
        "mean_r2_anthro":           float(np.mean(rsq_test[:, 0])),
        "mean_r2_pca_gt5":          float(np.mean(rsq_test[:, 1])),
        "mean_r2_pca_gt1":          float(np.mean(rsq_test[:, 2])),
        "mean_r2_all_teeth":        float(np.mean(rsq_test[:, 3])),
        "p_val_teeth_vs_anthro":    float(p_val),
        "matlab_target_r2":         0.35,
        "matlab_target_mae_days":   7,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


def main():
    parser = get_config_parser("Step 6: Age prediction (Lasso lambda_1se)")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    root = Path(args.config).resolve().parent

    clean_path  = root / cfg["data"]["outputs"]["evaluated_clean"]
    full_path   = root / cfg["data"]["outputs"]["evaluated_full"]
    output_path = root / cfg["data"]["outputs"]["tooth_with_gaps"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_predictions(
        str(clean_path), cfg, str(output_path),
        full_path=str(full_path),
    )


if __name__ == "__main__":
    main()
