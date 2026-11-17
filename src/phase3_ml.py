"""
src/phase3_ml.py
Módulo del Torneo de Modelos Predictivos y Calibración (Fase 3 / Unidades 4 y 5).
Responsable de Fase: Martín Pena (Machine Learning Engineer)
Integración Modular: Equipo EAyMR 2026 - UCA

Implementa:
1. Ingeniería de features respetando la partición congelada Train/Test (sin data leakage).
2. Regresión Logística Clásica (Statsmodels con Odds Ratios) y Regularizada (ElasticNet / Ridge / Lasso).
3. Clasificador Probabilístico Naive Bayes (GaussianNB) + auditoría de independencia + Calibración Isotónica.
4. Árbol de Decisión CART completo vs. Poda Cost-Complexity (ccp_alpha por CV).
5. Random Forest con RandomizedSearchCV + Curva de convergencia Out-Of-Bag (OOB).
6. Evaluación comparativa en Test (Precision, Recall, F1, AUC-ROC, PR-AUC, Brier Score, Reliability).
7. Importancia de variables MDI vs. Permutation Importance en Test + Ablación de DEBTINC_is_missing.
8. Serialización contractual hacia artifacts/best_classifier.joblib y artifacts/ml_metrics.json.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    BEST_CLASSIFIER_PATH,
    CV_FOLDS,
    ML_METRICS_PATH,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_BINARY,
    TEST_PATH,
    TRAIN_PATH,
    TREATMENT_VAR,
)

JOB_CATEGORIES = ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"]
BASE_MODEL_COLS = list(NUMERICAL_FEATURES) + ["DEBTINC_is_missing", TREATMENT_VAR]
INTERPRETABLE_SCALES = {"LOAN": 1e4, "MORTDUE": 1e4, "VALUE": 1e4, "CLAGE": 12}
INTERPRETABLE_UNITS = {
    "LOAN": "+ USD 10.000",
    "MORTDUE": "+ USD 10.000",
    "VALUE": "+ USD 10.000",
    "CLAGE": "+ 1 año",
}


def build_feature_matrix(df: pd.DataFrame, reference_columns: list[str] | None = None) -> pd.DataFrame:
    """Construye la matriz de 17 features manteniendo Mgr como categoría de referencia de JOB."""
    drop_cols = [c for c in ["BAD", "REASON"] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    if "JOB" in X.columns:
        X["JOB"] = pd.Categorical(X["JOB"], categories=JOB_CATEGORIES)
        X = pd.get_dummies(X, columns=["JOB"], drop_first=True, dtype=int)
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0)
    return X


def build_raw_preprocessor() -> ColumnTransformer:
    """Preprocesador puro de scikit-learn que acepta filas crudas de train.csv / Web App."""
    return ColumnTransformer(
        transformers=[
            ("num_bin", "passthrough", BASE_MODEL_COLS),
            (
                "job_ohe",
                OneHotEncoder(
                    categories=[JOB_CATEGORIES],
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=int,
                ),
                ["JOB"],
            ),
        ],
        remainder="drop",
    )


def load_phase3_data(
    train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga la partición oficial Train/Test sin fuga de datos y construye matrices de diseño."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = build_feature_matrix(train_df)
    X_test = build_feature_matrix(test_df, reference_columns=list(X_train.columns))
    y_train = train_df[TARGET_BINARY].astype(int)
    y_test = test_df[TARGET_BINARY].astype(int)
    return train_df, test_df, X_train, X_test, y_train, y_test


def _make_cv(cv_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)


def fit_logistic_suite(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Ajusta Logit de Statsmodels (con OR) y LogisticRegression ElasticNet + Ridge vs Lasso."""
    cv = _make_cv(cv_splits, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logit_sm = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0, maxiter=200)

        b = logit_sm.params.drop("const")
        se = logit_sm.bse.drop("const")
        k = np.array([INTERPRETABLE_SCALES.get(v, 1) for v in b.index], dtype=float)
        or_table = pd.DataFrame(
            {
                "Incremento": [INTERPRETABLE_UNITS.get(v, "+ 1 unidad") for v in b.index],
                "OR": np.exp(k * b),
                "IC_95_inf": np.exp(k * (b - 1.96 * se)),
                "IC_95_sup": np.exp(k * (b + 1.96 * se)),
                "p_valor": logit_sm.pvalues.drop("const"),
            },
            index=b.index,
        )

        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, random_state=random_state),
        )
        param_grid = {
            "logisticregression__C": [0.01, 0.1, 1, 10],
            "logisticregression__l1_ratio": [0.0, 0.5, 1.0],
        }
        grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        coefs = {}
        for label, pen in [("Ridge_L2", "l2"), ("Lasso_L1", "l1")]:
            est = make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty=pen, C=0.01, solver="saga", max_iter=5000, random_state=random_state),
            )
            est.fit(X_train, y_train)
            coefs[label] = est[-1].coef_[0]
        pen_comparison = pd.DataFrame(coefs, index=X_train.columns)

    return {
        "logit_sm": logit_sm,
        "pseudo_r2": float(logit_sm.prsquared),
        "llr_pvalue": float(logit_sm.llr_pvalue),
        "or_table": or_table,
        "best_model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_cv_auc": float(grid_search.best_score_),
        "cv_results": pd.DataFrame(grid_search.cv_results_),
        "pen_comparison": pen_comparison,
    }


def fit_naive_bayes_and_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Ajusta GaussianNB original y su versión calibrada por regresión isotónica (CV=5)."""
    cv = _make_cv(cv_splits, random_state)
    nb_raw = GaussianNB().fit(X_train, y_train)
    nb_calibrated = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=cv).fit(X_train, y_train)

    rows = []
    for name, model in [("Naive Bayes original", nb_raw), ("Naive Bayes calibrado", nb_calibrated)]:
        probs = model.predict_proba(X_test)[:, 1]
        obs_freq, pred_mean = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        rows.append(
            {
                "Modelo": name,
                "Prob_media": float(probs.mean()),
                "Error_calibracion": float(np.mean(np.abs(obs_freq - pred_mean))),
                "Brier": float(brier_score_loss(y_test, probs)),
                "AUC_ROC": float(roc_auc_score(y_test, probs)),
                "Recall_05": float(recall_score(y_test, (probs >= 0.5).astype(int))),
                "curve_pred": pred_mean.tolist(),
                "curve_obs": obs_freq.tolist(),
            }
        )

    num_cols = list(NUMERICAL_FEATURES)
    corr_paid = X_train.loc[y_train == 0, num_cols].corr()
    corr_default = X_train.loc[y_train == 1, num_cols].corr()

    return {
        "nb_raw": nb_raw,
        "nb_calibrated": nb_calibrated,
        "calibration_comparison": pd.DataFrame(rows).drop(columns=["curve_pred", "curve_obs"]).set_index("Modelo"),
        "calibration_curves": rows,
        "corr_paid": corr_paid,
        "corr_default": corr_default,
    }


def fit_decision_tree_pruning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Ajusta árbol CART completo vs. árbol podado por cost-complexity (ccp_alpha óptimo en CV)."""
    cv = _make_cv(cv_splits, random_state)
    tree_full = DecisionTreeClassifier(random_state=random_state).fit(X_train, y_train)

    path = DecisionTreeClassifier(random_state=random_state).cost_complexity_pruning_path(X_train, y_train)
    alphas = np.unique(np.quantile(np.unique(path.ccp_alphas), np.linspace(0, 0.98, 30)))
    cv_aucs = [
        float(
            cross_val_score(
                DecisionTreeClassifier(random_state=random_state, ccp_alpha=float(a)),
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            ).mean()
        )
        for a in alphas
    ]
    pruning_curve = pd.DataFrame({"alpha": alphas, "auc_cv": cv_aucs})
    best_alpha = float(pruning_curve.loc[pruning_curve["auc_cv"].idxmax(), "alpha"])
    tree_pruned = DecisionTreeClassifier(random_state=random_state, ccp_alpha=best_alpha).fit(X_train, y_train)

    return {
        "tree_full": tree_full,
        "tree_pruned": tree_pruned,
        "best_alpha": best_alpha,
        "pruning_curve": pruning_curve,
        "full_stats": {
            "leaves": int(tree_full.get_n_leaves()),
            "depth": int(tree_full.get_depth()),
            "auc_train": float(roc_auc_score(y_train, tree_full.predict_proba(X_train)[:, 1])),
            "auc_test": float(roc_auc_score(y_test, tree_full.predict_proba(X_test)[:, 1])),
        },
        "pruned_stats": {
            "leaves": int(tree_pruned.get_n_leaves()),
            "depth": int(tree_pruned.get_depth()),
            "auc_train": float(roc_auc_score(y_train, tree_pruned.predict_proba(X_train)[:, 1])),
            "auc_test": float(roc_auc_score(y_test, tree_pruned.predict_proba(X_test)[:, 1])),
        },
    }


def fit_random_forest_tournament(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    n_iter: int = 20,
) -> dict[str, Any]:
    """Ajusta Random Forest con RandomizedSearchCV (CV=5) y calcula curva de convergencia OOB."""
    cv = _make_cv(cv_splits, random_state)
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 8, 12, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=1),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_

    # Convergencia Out-Of-Bag con los hiperparámetros ganadores
    base_params = {k: v for k, v in search.best_params_.items() if k != "n_estimators"}
    oob_rf = RandomForestClassifier(warm_start=True, oob_score=True, random_state=random_state, n_jobs=-1, **base_params)
    oob_rows = []
    for n in [25, 50, 100, 200, 300, 500]:
        oob_rf.set_params(n_estimators=n).fit(X_train, y_train)
        oob_rows.append(
            {
                "n_arboles": n,
                "error_oob": float(1.0 - oob_rf.oob_score_),
                "auc_oob": float(roc_auc_score(y_train, oob_rf.oob_decision_function_[:, 1])),
            }
        )

    return {
        "best_model": best_rf,
        "best_params": search.best_params_,
        "best_cv_auc": float(search.best_score_),
        "cv_results": pd.DataFrame(search.cv_results_),
        "oob_curve": pd.DataFrame(oob_rows),
    }


def evaluate_models_test(
    models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    """Evalúa todas las familias sobre Test (métricas globales, curvas ROC, PR y calibración)."""
    metrics_rows = []
    roc_data = {}
    pr_data = {}
    calib_data = {}

    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        y_hat = (probs >= 0.5).astype(int)
        metrics_rows.append(
            {
                "Modelo": name,
                "Precision": float(precision_score(y_test, y_hat)),
                "Recall": float(recall_score(y_test, y_hat)),
                "F1_macro": float(f1_score(y_test, y_hat, average="macro")),
                "F1_weighted": float(f1_score(y_test, y_hat, average="weighted")),
                "AUC_ROC": float(roc_auc_score(y_test, probs)),
                "PR_AUC": float(average_precision_score(y_test, probs)),
                "Brier_Score": float(brier_score_loss(y_test, probs)),
                "Mean_Pred_Prob": float(probs.mean()),
            }
        )
        fpr, tpr, _ = roc_curve(y_test, probs)
        prec, rec, _ = precision_recall_curve(y_test, probs)
        obs_cal, pred_cal = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        pr_data[name] = {"precision": prec.tolist(), "recall": rec.tolist()}
        calib_data[name] = {
            "pred": pred_cal.tolist(),
            "obs": obs_cal.tolist(),
            "ece_quantile": float(np.mean(np.abs(obs_cal - pred_cal))),
        }

    metrics_df = pd.DataFrame(metrics_rows).set_index("Modelo")
    return {
        "metrics_table": metrics_df,
        "roc_curves": roc_data,
        "pr_curves": pr_data,
        "calibration_curves": calib_data,
    }


def compute_feature_importance_comparison(
    rf_model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = RANDOM_STATE,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Compara MDI (impureza en Train) vs. Permutation Importance (caída AUC en Test)."""
    perm = permutation_importance(
        rf_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "MDI": rf_model.feature_importances_,
            "Permutacion": perm.importances_mean,
            "Permutacion_SD": perm.importances_std,
            "Valores_distintos": X_train.nunique().values,
        },
        index=X_train.columns,
    )
    table["Rank_MDI"] = table["MDI"].rank(ascending=False).astype(int)
    table["Rank_Perm"] = table["Permutacion"].rank(ascending=False).astype(int)
    return table.sort_values("Permutacion", ascending=False)


def run_debtinc_missing_ablation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_rf_params: dict[str, Any],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evalúa el impacto de remover DEBTINC_is_missing y DEBTINC en Random Forest vs. Logística."""
    scenarios = [
        ("Todas las variables", []),
        ("Sin el indicador", ["DEBTINC_is_missing"]),
        ("Sin el indicador ni DEBTINC", ["DEBTINC_is_missing", "DEBTINC"]),
    ]
    rows = []
    for name, drop_cols in scenarios:
        Xtr = X_train.drop(columns=drop_cols)
        Xte = X_test.drop(columns=drop_cols)
        rf = RandomForestClassifier(**best_rf_params, random_state=random_state, n_jobs=-1).fit(Xtr, y_train)
        lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=random_state)).fit(Xtr, y_train)
        rows.append(
            {
                "Escenario": name,
                "AUC_Random_Forest": float(roc_auc_score(y_test, rf.predict_proba(Xte)[:, 1])),
                "AUC_Logistica": float(roc_auc_score(y_test, lg.predict_proba(Xte)[:, 1])),
            }
        )
    return pd.DataFrame(rows).set_index("Escenario")


def build_deployable_classifier(
    train_df: pd.DataFrame, best_rf_params: dict[str, Any], random_state: int = RANDOM_STATE
) -> Pipeline:
    """Entrena el Pipeline completo (ColumnTransformer + RandomForest) sobre train_df crudo para producción."""
    preprocessor = build_raw_preprocessor()
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**best_rf_params, random_state=random_state, n_jobs=-1)),
        ]
    )
    y_train = train_df[TARGET_BINARY].astype(int)
    pipeline.fit(train_df, y_train)
    return pipeline


def run_phase3_pipeline(save: bool = True) -> dict[str, Any]:
    """Orquesta todo el Torneo ML (Fase 3) y exporta artefactos hacia artifacts/."""
    print("=" * 80)
    print("INICIANDO FASE 3: TORNEO DE MODELOS PREDICTIVOS Y CALIBRACIÓN")
    print("=" * 80)

    train_df, test_df, X_train, X_test, y_train, y_test = load_phase3_data()
    print(f"[ML] Train: {X_train.shape} (default={y_train.mean():.2%}) | Test: {X_test.shape} (default={y_test.mean():.2%})")

    logit_res = fit_logistic_suite(X_train, y_train)
    print(f"[ML] Logit SM Pseudo-R2: {logit_res['pseudo_r2']:.4f} | ElasticNet CV AUC: {logit_res['best_cv_auc']:.4f}")

    nb_res = fit_naive_bayes_and_calibration(X_train, y_train, X_test, y_test)
    tree_res = fit_decision_tree_pruning(X_train, y_train, X_test, y_test)
    print(
        f"[ML] CART sin poda: {tree_res['full_stats']['leaves']} hojas (AUC test={tree_res['full_stats']['auc_test']:.4f}) -> "
        f"Podado: {tree_res['pruned_stats']['leaves']} hojas (AUC test={tree_res['pruned_stats']['auc_test']:.4f})"
    )

    rf_res = fit_random_forest_tournament(X_train, y_train)
    print(f"[ML] Random Forest CV AUC: {rf_res['best_cv_auc']:.4f} | Params: {rf_res['best_params']}")

    models = {
        "Logística (Elastic Net)": logit_res["best_model"],
        "Naive Bayes": nb_res["nb_raw"],
        "Naive Bayes Calibrado": nb_res["nb_calibrated"],
        "Árbol sin poda": tree_res["tree_full"],
        "Árbol podado": tree_res["tree_pruned"],
        "Random Forest": rf_res["best_model"],
    }
    eval_res = evaluate_models_test(models, X_test, y_test)
    imp_df = compute_feature_importance_comparison(rf_res["best_model"], X_train, X_test, y_test)
    ablation_df = run_debtinc_missing_ablation(X_train, y_train, X_test, y_test, rf_res["best_params"])

    deploy_pipeline = build_deployable_classifier(train_df, rf_res["best_params"])

    summary_json = {
        "dataset_info": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": int(X_train.shape[1]),
            "feature_names": list(X_train.columns),
            "test_default_rate": float(y_test.mean()),
        },
        "winner": {
            "model_name": "Random Forest",
            "best_params": {k: (None if v is None else v) for k, v in rf_res["best_params"].items()},
            "cv_roc_auc": float(rf_res["best_cv_auc"]),
            "test_roc_auc": float(eval_res["metrics_table"].loc["Random Forest", "AUC_ROC"]),
            "test_pr_auc": float(eval_res["metrics_table"].loc["Random Forest", "PR_AUC"]),
            "test_brier": float(eval_res["metrics_table"].loc["Random Forest", "Brier_Score"]),
        },
        "logistic_statsmodels": {
            "pseudo_r2_mcfadden": logit_res["pseudo_r2"],
            "llr_pvalue": logit_res["llr_pvalue"],
            "elasticnet_best_params": logit_res["best_params"],
            "elasticnet_cv_auc": logit_res["best_cv_auc"],
        },
        "naive_bayes_calibration": nb_res["calibration_curves"],
        "decision_tree_pruning": {
            "best_ccp_alpha": tree_res["best_alpha"],
            "full_tree": tree_res["full_stats"],
            "pruned_tree": tree_res["pruned_stats"],
        },
        "oob_convergence": rf_res["oob_curve"].to_dict(orient="records"),
        "model_comparison_test": eval_res["metrics_table"].reset_index().to_dict(orient="records"),
        "calibration_curves": eval_res["calibration_curves"],
        "roc_curves": eval_res["roc_curves"],
        "pr_curves": eval_res["pr_curves"],
        "feature_importance": imp_df.reset_index().rename(columns={"index": "Feature"}).to_dict(orient="records"),
        "debtinc_ablation": ablation_df.reset_index().to_dict(orient="records"),
    }

    if save:
        BEST_CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(deploy_pipeline, BEST_CLASSIFIER_PATH)
        with open(ML_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=4, ensure_ascii=False)
        print(f"[ML] Modelo serializado en: {BEST_CLASSIFIER_PATH}")
        print(f"[ML] Métricas exportadas en: {ML_METRICS_PATH}")

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "logit_res": logit_res,
        "nb_res": nb_res,
        "tree_res": tree_res,
        "rf_res": rf_res,
        "models": models,
        "eval_res": eval_res,
        "importance_df": imp_df,
        "ablation_df": ablation_df,
        "deploy_pipeline": deploy_pipeline,
        "summary": summary_json,
    }


if __name__ == "__main__":
    run_phase3_pipeline(save=True)
