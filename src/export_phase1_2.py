"""
src/export_phase1_2.py
Exporta los artefactos serializados de las Fases 1 y 2 hacia artifacts/
para consumo directo de la Web App (Fase 5) y del equipo Scrum.
"""

import json

import joblib

from src.config import (
    ARTIFACTS_DIR,
    PCA_PIPELINE_PATH,
    PHASE1_2_SUMMARY_PATH,
    TARGET_CONTINUOUS,
)
from src.phase1_inference import (
    analyze_missing_mechanism,
    compute_multidimensional_eda,
    load_phase1_datasets,
    run_anova_and_diagnostics,
    run_hypothesis_contrasts,
)
from src.phase2_econometrics import (
    audit_gauss_markov,
    calculate_vif,
    compare_robust_regressors,
    detect_influential_points,
    fit_ols_reference,
    load_phase2_splits,
    run_pca_no_leakage,
)


def export_phase1_2_artifacts() -> tuple[str, str]:
    """Ejecuta Fases 1 y 2 y serializa artefactos en joblib y json."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- FASE 1 ---
    df_raw, train_df = load_phase1_datasets()
    missing_res = analyze_missing_mechanism(df_raw)
    eda_res = compute_multidimensional_eda(train_df)
    contrasts_res = run_hypothesis_contrasts(df_raw, alpha=0.05)
    anova_res = run_anova_and_diagnostics(train_df)

    # --- FASE 2 ---
    tr_df, te_df, pred_cols = load_phase2_splits()
    ols_model = fit_ols_reference(tr_df[pred_cols], tr_df[TARGET_CONTINUOUS])
    gm_audit = audit_gauss_markov(ols_model)
    infl_res = detect_influential_points(ols_model, tr_df[pred_cols])
    robust_comp = compare_robust_regressors(
        tr_df[pred_cols], tr_df[TARGET_CONTINUOUS],
        te_df[pred_cols], te_df[TARGET_CONTINUOUS],
        ols_model
    )
    vif_df = calculate_vif(tr_df[pred_cols])
    pca_res = run_pca_no_leakage(tr_df, te_df, pred_cols)

    # 1. Serializar PCA Pipeline (joblib) para Módulo 1 de la Web App
    pca_bundle = {
        "feature_cols": pca_res["feature_cols"],
        "scaler": pca_res["scaler"],
        "pca": pca_res["pca"],
        "eigenvalues": pca_res["eigenvalues"].tolist(),
        "var_exp_ratio": pca_res["var_exp_ratio"].tolist(),
        "cum_var_exp": pca_res["cum_var_exp"].tolist(),
        "kaiser_retained_count": pca_res["kaiser_retained_count"],
        "loadings": pca_res["loadings"].to_dict(),
        "X_train_pca_2d": pca_res["X_train_pca"][:, :2].tolist(),
        "X_test_pca_2d": pca_res["X_test_pca"][:, :2].tolist(),
    }
    joblib.dump(pca_bundle, PCA_PIPELINE_PATH)

    # 2. Serializar Resumen Estadístico y Econométrico (JSON)
    summary_payload = {
        "fase_1": {
            "missing_mechanism": {
                "is_mcar": missing_res["is_mcar"],
                "odds_ratio_bad_missing_debtinc": missing_res["odds_ratio_bad"],
                "missing_table": missing_res["missing_table"].reset_index().rename(
                    columns={"index": "Variable"}
                ).to_dict(orient="records")
            },
            "eda_descriptivos": eda_res["descriptivos"].to_dict(orient="records"),
            "contrastes": contrasts_res,
            "anova": {
                "diagnostico_grupos": anova_res["diagnostico_grupos"].to_dict(orient="records"),
                "anova_clasico": anova_res["anova_clasico"],
                "levene_test": anova_res["levene_test"],
                "kruskal_wallis": anova_res["kruskal_wallis"],
                "welch_anova": anova_res["welch_anova"],
                "bonferroni_welch_posthoc": anova_res["bonferroni_welch_posthoc"].to_dict(orient="records")
            }
        },
        "fase_2": {
            "ols_reference": {
                "r2": float(ols_model.rsquared),
                "r2_adj": float(ols_model.rsquared_adj),
                "f_stat": float(ols_model.fvalue),
                "f_pvalue": float(ols_model.f_pvalue),
                "params": {k: float(v) for k, v in ols_model.params.items()},
                "pvalues": {k: float(v) for k, v in ols_model.pvalues.items()}
            },
            "gauss_markov": gm_audit,
            "influential_points": {
                "n": infl_res["n"],
                "p": infl_res["p"],
                "thresh_leverage": infl_res["thresh_leverage"],
                "thresh_cook_4n": infl_res["thresh_cook_4n"],
                "count_outliers_resid": infl_res["count_outliers_resid"],
                "count_high_leverage": infl_res["count_high_leverage"],
                "count_high_cooks_4n": infl_res["count_high_cooks_4n"],
                "count_extreme_cooks_half": infl_res["count_extreme_cooks_half"],
                "max_cooks_d": infl_res["max_cooks_d"]
            },
            "robust_comparison": robust_comp.to_dict(orient="records"),
            "vif": vif_df.to_dict(orient="records"),
            "pca": {
                "kaiser_retained_count": pca_res["kaiser_retained_count"],
                "eigenvalues": pca_res["eigenvalues"].tolist(),
                "var_exp_ratio": pca_res["var_exp_ratio"].tolist(),
                "cum_var_exp": pca_res["cum_var_exp"].tolist()
            }
        }
    }

    with open(PHASE1_2_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=4, ensure_ascii=False)

    print(f"[EXPORT] Bundle PCA guardado en: {PCA_PIPELINE_PATH}")
    print(f"[EXPORT] Resumen Fases 1 y 2 guardado en: {PHASE1_2_SUMMARY_PATH}")
    return str(PCA_PIPELINE_PATH), str(PHASE1_2_SUMMARY_PATH)


if __name__ == "__main__":
    export_phase1_2_artifacts()
