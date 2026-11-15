"""
src/phase1_inference.py
Módulo de Inferencia Estadística Clásica, EDA Multidimensional, Contrastes y ANOVA (Fase 1)
Cátedra: Estadística Aplicada y Modelos de Regresión (EAyMR 2026)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src.config import (
    ANOVA_FACTOR,
    ANOVA_VARIABLE,
    DATA_RAW_PATH,
    NUMERICAL_FEATURES,
    TARGET_BINARY,
    TARGET_CONTINUOUS,
    TRAIN_PATH,
)


def load_phase1_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga el dataset crudo (para auditoría de pérdida MCAR/MAR/MNAR)
    y la partición de entrenamiento procesada (para análisis inferencial libre de fuga).
    """
    df_raw = pd.read_csv(DATA_RAW_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    return df_raw, train_df


def analyze_missing_mechanism(df_raw: pd.DataFrame) -> dict:
    """
    Analiza el mecanismo de pérdida de datos para clasificar entre MCAR, MAR y MNAR.
    Ajusta una regresión logística modelando el indicador de dato faltante en DEBTINC
    en función de LOAN, VALUE y el estado crediticio BAD.
    """
    missing_summary = df_raw.isnull().sum()
    missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100
    missing_table = pd.DataFrame({
        "Nulos": missing_summary,
        "Porcentaje (%)": missing_pct
    }).sort_values("Porcentaje (%)", ascending=False)

    df_temp = df_raw.copy()
    df_temp["DEBTINC_missing"] = df_temp["DEBTINC"].isnull().astype(int)
    logit_model = sm.Logit.from_formula(
        f"DEBTINC_missing ~ {TARGET_CONTINUOUS} + VALUE + {TARGET_BINARY}",
        data=df_temp
    ).fit(disp=False)

    is_mcar = bool(logit_model.pvalues[TARGET_BINARY] >= 0.05)

    return {
        "missing_table": missing_table,
        "logit_debtinc_missing": logit_model,
        "is_mcar": is_mcar,
        "odds_ratio_bad": float(np.exp(logit_model.params[TARGET_BINARY]))
    }


def compute_multidimensional_eda(df: pd.DataFrame) -> dict:
    """
    Calcula estadísticas descriptivas univariadas (incluyendo asimetría y curtosis)
    y matrices de correlación paramétrica (Pearson) y no paramétrica (Spearman).
    """
    desc_rows = []
    for col in NUMERICAL_FEATURES:
        vals = df[col].dropna()
        desc_rows.append({
            "Variable": col,
            "N": len(vals),
            "Media": vals.mean(),
            "Mediana": vals.median(),
            "Desv_Est": vals.std(ddof=1),
            "Asimetria": stats.skew(vals),
            "Curtosis": stats.kurtosis(vals),
            "IQR": vals.quantile(0.75) - vals.quantile(0.25)
        })
    desc_table = pd.DataFrame(desc_rows)
    corr_pearson = df[NUMERICAL_FEATURES].corr(method="pearson")
    corr_spearman = df[NUMERICAL_FEATURES].corr(method="spearman")

    return {
        "descriptivos": desc_table,
        "corr_pearson": corr_pearson,
        "corr_spearman": corr_spearman
    }


def _welch_ci_and_cohen_d(g_a: pd.Series, g_b: pd.Series) -> tuple[tuple[float, float], float]:
    """Calcula el IC 95% de Welch para (mean(g_b) - mean(g_a)) y el tamaño del efecto d de Cohen."""
    n_a, n_b = len(g_a), len(g_b)
    var_a, var_b = g_a.var(ddof=1), g_b.var(ddof=1)
    diff_mean = g_b.mean() - g_a.mean()

    se_diff = np.sqrt(var_a / n_a + var_b / n_b)
    dof = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    )
    t_crit = stats.t.ppf(0.975, dof)
    ci95 = (float(diff_mean - t_crit * se_diff), float(diff_mean + t_crit * se_diff))

    s_pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohen_d = float(diff_mean / s_pooled)
    return ci95, cohen_d


def run_hypothesis_contrasts(df_raw: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Ejecuta dos contrastes de hipótesis formales con relevancia directa para el negocio:
    1. LOAN según BAD (Cumplidor vs Moroso) - Bilateral.
    2. DEBTINC según REASON (DebtCon vs HomeImp) - Unilateral (DebtCon > HomeImp).
    """
    contrasts = {}

    # Contraste 1: LOAN según BAD
    g0 = df_raw[df_raw[TARGET_BINARY] == 0][TARGET_CONTINUOUS].dropna()
    g1 = df_raw[df_raw[TARGET_BINARY] == 1][TARGET_CONTINUOUS].dropna()
    t_stat1, p_welch1 = stats.ttest_ind(g0, g1, equal_var=False)
    u_stat1, p_mwu1 = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    ci1, d1 = _welch_ci_and_cohen_d(g0, g1)

    contrasts["contraste_1"] = {
        "titulo": f"{TARGET_CONTINUOUS} según {TARGET_BINARY} (0=Cumplidor vs 1=Moroso)",
        "alpha": alpha,
        "mean_0": float(g0.mean()),
        "std_0": float(g0.std(ddof=1)),
        "mean_1": float(g1.mean()),
        "std_1": float(g1.std(ddof=1)),
        "diff_mean": float(g1.mean() - g0.mean()),
        "diff_ci95": ci1,
        "t_stat": float(t_stat1),
        "p_value_welch": float(p_welch1),
        "u_stat": float(u_stat1),
        "p_value_mwu": float(p_mwu1),
        "cohen_d": d1,
        "reject_h0": bool(p_welch1 < alpha)
    }

    # Contraste 2: DEBTINC según REASON (DebtCon vs HomeImp)
    sub2 = df_raw.dropna(subset=["DEBTINC", "REASON"])
    g_hi = sub2[sub2["REASON"] == "HomeImp"]["DEBTINC"]
    g_dc = sub2[sub2["REASON"] == "DebtCon"]["DEBTINC"]
    t_stat2, p_welch2 = stats.ttest_ind(g_dc, g_hi, equal_var=False, alternative="greater")
    u_stat2, p_mwu2 = stats.mannwhitneyu(g_dc, g_hi, alternative="greater")
    ci2, d2 = _welch_ci_and_cohen_d(g_hi, g_dc)

    contrasts["contraste_2"] = {
        "titulo": "DEBTINC según REASON (DebtCon > HomeImp)",
        "alpha": alpha,
        "mean_HomeImp": float(g_hi.mean()),
        "std_HomeImp": float(g_hi.std(ddof=1)),
        "mean_DebtCon": float(g_dc.mean()),
        "std_DebtCon": float(g_dc.std(ddof=1)),
        "diff_mean": float(g_dc.mean() - g_hi.mean()),
        "diff_ci95": ci2,
        "t_stat": float(t_stat2),
        "p_value_welch": float(p_welch2),
        "u_stat": float(u_stat2),
        "p_value_mwu": float(p_mwu2),
        "cohen_d": d2,
        "reject_h0": bool(p_welch2 < alpha)
    }

    return contrasts


def run_anova_and_diagnostics(
    df: pd.DataFrame,
    continuous_var: str = ANOVA_VARIABLE,
    group_var: str = ANOVA_FACTOR,
    alpha: float = 0.05
) -> dict:
    """
    Ejecuta el ANOVA de un factor con diagnóstico riguroso de supuestos:
    - Normalidad en muestras masivas: Shapiro-Wilk + asimetría [-1, 1] + curtosis.
    - Homocedasticidad: Test de Levene (centrado en la mediana).
    - Remediación heterocedástica: Welch's ANOVA (F*) y Kruskal-Wallis.
    - Comparaciones múltiples post-hoc: Tukey HSD y pares Welch con corrección de Bonferroni.
    """
    sub = df.dropna(subset=[continuous_var, group_var]).copy()
    groups = {name: grp[continuous_var].values for name, grp in sub.groupby(group_var)}

    diag_rows = []
    for name, vals in sorted(groups.items()):
        sample_sw = vals if len(vals) <= 5000 else vals[:5000]
        sw_stat, sw_p = stats.shapiro(sample_sw)
        skew_val = stats.skew(vals)
        diag_rows.append({
            "Grupo": name,
            "N": len(vals),
            "Media": float(np.mean(vals)),
            "Desv_Est": float(np.std(vals, ddof=1)),
            "Asimetria": float(skew_val),
            "Asimetria_OK": bool(-1.0 <= skew_val <= 1.0),
            "Curtosis": float(stats.kurtosis(vals)),
            "Shapiro_stat": float(sw_stat),
            "Shapiro_p": float(sw_p)
        })
    diag_table = pd.DataFrame(diag_rows)

    # ANOVA clásico de Fisher
    f_stat, classic_p = stats.f_oneway(*groups.values())

    # Levene (Brown-Forsythe)
    lev_stat, lev_p = stats.levene(*groups.values(), center="median")

    # Kruskal-Wallis
    kw_stat, kw_p = stats.kruskal(*groups.values())

    # Welch's ANOVA (1951)
    means = [np.mean(v) for v in groups.values()]
    variances = [np.var(v, ddof=1) for v in groups.values()]
    ns = [len(v) for v in groups.values()]
    weights = [n / var for n, var in zip(ns, variances)]
    sum_w = sum(weights)
    w_mean = sum(w * m for w, m in zip(weights, means)) / sum_w

    k = len(groups)
    f_num = sum(w * (m - w_mean) ** 2 for w, m in zip(weights, means)) / (k - 1)
    denom_sum = sum((1.0 - w / sum_w) ** 2 / (n - 1) for w, n in zip(weights, ns))
    f_den = 1.0 + (2.0 * (k - 2.0) / (k ** 2 - 1.0)) * denom_sum
    f_welch = float(f_num / f_den)
    df1_welch = k - 1
    df2_welch = float((k ** 2 - 1.0) / (3.0 * denom_sum))
    p_welch = float(1.0 - stats.f.cdf(f_welch, df1_welch, df2_welch))

    # Post-hoc 1: Tukey HSD
    tukey = pairwise_tukeyhsd(endog=sub[continuous_var], groups=sub[group_var], alpha=alpha)

    # Post-hoc 2: Pares Welch t-test con corrección de Bonferroni (para heterocedasticidad)
    group_names = sorted(groups.keys())
    n_pairs = k * (k - 1) // 2
    bonf_rows = []
    for i in range(k):
        for j in range(i + 1, k):
            g1_name, g2_name = group_names[i], group_names[j]
            t_p, p_raw = stats.ttest_ind(groups[g1_name], groups[g2_name], equal_var=False)
            p_bonf = min(float(p_raw * n_pairs), 1.0)
            bonf_rows.append({
                "Grupo_1": g1_name,
                "Grupo_2": g2_name,
                "Diff_Medias": float(np.mean(groups[g2_name]) - np.mean(groups[g1_name])),
                "Welch_t": float(t_p),
                "p_raw": float(p_raw),
                "p_bonferroni": p_bonf,
                "Rechaza_H0": bool(p_bonf < alpha)
            })
    bonferroni_df = pd.DataFrame(bonf_rows)

    return {
        "diagnostico_grupos": diag_table,
        "anova_clasico": {"F_stat": float(f_stat), "p_value": float(classic_p)},
        "levene_test": {"W_stat": float(lev_stat), "p_value": float(lev_p), "homocedastico": bool(lev_p >= alpha)},
        "kruskal_wallis": {"H_stat": float(kw_stat), "p_value": float(kw_p)},
        "welch_anova": {
            "F_welch": f_welch,
            "df1": df1_welch,
            "df2": df2_welch,
            "p_value": p_welch
        },
        "tukey_posthoc": tukey,
        "bonferroni_welch_posthoc": bonferroni_df
    }
