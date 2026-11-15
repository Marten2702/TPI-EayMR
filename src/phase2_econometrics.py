"""
src/phase2_econometrics.py
Módulo de Diagnóstico Econométrico de Gauss-Markov, Puntos Influyentes, VIF y PCA sin Leakage (Fase 2)
Cátedra: Estadística Aplicada y Modelos de Regresión (EAyMR 2026)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from src.config import (
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_CONTINUOUS,
    TEST_PATH,
    TRAIN_PATH,
)


def load_phase2_splits() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Carga las particiones oficiales limpias train.csv y test.csv
    generadas por data_pipeline.py, evitando cualquier fuga de datos.
    """
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    predictor_cols = [c for c in NUMERICAL_FEATURES if c != TARGET_CONTINUOUS]
    return train_df, test_df, predictor_cols


def fit_ols_reference(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Ajusta el modelo lineal clásico de referencia por Mínimos Cuadrados Ordinarios (MCO)
    exclusivamente sobre la partición de entrenamiento (Train).
    """
    X_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_const).fit()
    return model


def audit_gauss_markov(model) -> dict:
    """
    Auditoría exhaustiva de supuestos de Gauss-Markov:
    1. Homocedasticidad: Test de Breusch-Pagan y Test de White.
    2. Normalidad de residuos: Test de Jarque-Bera (asimetría y curtosis).
    3. Independencia de errores: Estadístico de Durbin-Watson.
    """
    bp_lm, bp_pval, bp_f, bp_fpval = het_breuschpagan(model.resid, model.model.exog)
    wh_lm, wh_pval, wh_f, wh_fpval = het_white(model.resid, model.model.exog)
    jb_stat, jb_pval, skew, kurt = jarque_bera(model.resid)
    dw_stat = durbin_watson(model.resid)

    return {
        "breusch_pagan": {
            "lm_stat": float(bp_lm),
            "p_value": float(bp_pval),
            "f_stat": float(bp_f),
            "f_pval": float(bp_fpval),
            "is_homoscedastic": bool(bp_pval >= 0.05)
        },
        "white_test": {
            "lm_stat": float(wh_lm),
            "p_value": float(wh_pval),
            "f_stat": float(wh_f),
            "f_pval": float(wh_fpval),
            "is_homoscedastic": bool(wh_pval >= 0.05)
        },
        "jarque_bera": {
            "jb_stat": float(jb_stat),
            "p_value": float(jb_pval),
            "skewness": float(skew),
            "kurtosis": float(kurt),
            "is_normal": bool(jb_pval >= 0.05)
        },
        "durbin_watson": {
            "dw_stat": float(dw_stat),
            "is_independent": bool(1.5 <= dw_stat <= 2.5)
        }
    }


def detect_influential_points(model, X_train: pd.DataFrame) -> dict:
    """
    Descompone analíticamente el impacto de observaciones atípicas (formulación vectorial exacta):
    - Residuos estudentizados externamente (|r_i| > 3).
    - Apalancamiento (Leverage h_ii > 2(p+1)/n).
    - Distancia de Cook (D_i > 4/n y D_i > 0.5).
    """
    n = len(X_train)
    p = X_train.shape[1]
    e = model.resid.values
    mse = model.mse_resid

    # Diagonal de la matriz Hat via factorización QR
    X_mat = sm.add_constant(X_train).values
    Q, _ = np.linalg.qr(X_mat)
    leverage = np.sum(Q ** 2, axis=1)

    # Distancia de Cook
    denom_lev = np.maximum((1.0 - leverage) ** 2, 1e-12)
    cooks_d = (e ** 2 / ((p + 1) * mse)) * (leverage / denom_lev)

    # Residuos estudentizados externamente (leave-one-out MSE)
    df_e = n - p - 1
    mse_i = (df_e * mse - (e ** 2 / np.maximum(1.0 - leverage, 1e-12))) / (df_e - 1)
    mse_i = np.maximum(mse_i, 1e-12)
    stud_resids = e / (np.sqrt(mse_i) * np.sqrt(np.maximum(1.0 - leverage, 1e-12)))

    thresh_cook_4n = 4.0 / n
    thresh_cook_half = 0.5
    thresh_lev = 2.0 * (p + 1) / n

    outliers_resids = np.where(np.abs(stud_resids) > 3.0)[0]
    high_leverage = np.where(leverage > thresh_lev)[0]
    high_cooks_4n = np.where(cooks_d > thresh_cook_4n)[0]
    extreme_cooks_half = np.where(cooks_d > thresh_cook_half)[0]

    return {
        "n": n,
        "p": p,
        "thresh_leverage": float(thresh_lev),
        "thresh_cook_4n": float(thresh_cook_4n),
        "count_outliers_resid": len(outliers_resids),
        "count_high_leverage": len(high_leverage),
        "count_high_cooks_4n": len(high_cooks_4n),
        "count_extreme_cooks_half": len(extreme_cooks_half),
        "max_cooks_d": float(np.max(cooks_d)),
        "cooks_distance": cooks_d,
        "leverage": leverage,
        "stud_resids": stud_resids
    }


def compare_robust_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ols_model
) -> pd.DataFrame:
    """
    Compara el desempeño en Test (RMSE y MAE) entre MCO clásico y
    regresores robustos frente a puntos influyentes (Huber y RANSAC).
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    pred_ols = ols_model.predict(sm.add_constant(X_test))

    huber = HuberRegressor(max_iter=2000).fit(X_tr_s, y_train)
    pred_huber = huber.predict(X_te_s)

    ransac = RANSACRegressor(random_state=RANDOM_STATE).fit(X_tr_s, y_train)
    pred_ransac = ransac.predict(X_te_s)

    rows = []
    for name, preds in [("MCO (OLS)", pred_ols), ("Huber Robusto", pred_huber), ("RANSAC Robusto", pred_ransac)]:
        rows.append({
            "Modelo": name,
            "Test_RMSE": float(root_mean_squared_error(y_test, preds)),
            "Test_MAE": float(mean_absolute_error(y_test, preds))
        })
    return pd.DataFrame(rows)


def calculate_vif(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los Factores de Inflación de Varianza (VIF_j = 1 / (1 - R_j^2))
    sobre el conjunto de entrenamiento.
    """
    X_const = sm.add_constant(X_train)
    vif_rows = []
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        vif_val = float(variance_inflation_factor(X_const.values, i))
        vif_rows.append({
            "Variable": col,
            "VIF": vif_val,
            "Alerta_VIF_5": bool(vif_val > 5.0),
            "Alerta_VIF_10": bool(vif_val > 10.0)
        })
    return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def run_pca_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str] | None = None) -> dict:
    """
    Ejecuta PCA bajo regla estricta anti-leakage:
    - StandardScaler ajustado EXCLUSIVAMENTE sobre train_df.
    - Descomposición SVD/PCA ajustada EXCLUSIVAMENTE sobre train_df escalado.
    - Rotación aplicada ciegamente sobre test_df.
    """
    if feature_cols is None:
        feature_cols = [c for c in NUMERICAL_FEATURES if c != TARGET_CONTINUOUS]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    eigenvalues = pca.explained_variance_
    var_exp_ratio = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp_ratio)
    kaiser_retained = int(np.sum(eigenvalues > 1.0))

    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(eigenvalues),
        columns=[f"PC{i + 1}" for i in range(len(eigenvalues))],
        index=feature_cols
    )

    return {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "pca": pca,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "eigenvalues": eigenvalues,
        "var_exp_ratio": var_exp_ratio,
        "cum_var_exp": cum_var_exp,
        "kaiser_retained_count": kaiser_retained,
        "loadings": loadings
    }
