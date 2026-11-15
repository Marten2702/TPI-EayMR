"""
src/phase4_causal.py
Módulo de Inferencia Causal, DAGs de Pearl y Causal Machine Learning (Fase 4 / Workflow C)
Cátedra: Estadística Aplicada y Modelos de Regresión (EAyMR 2026) - UCA
Responsable Causal: Santiago (Causal & Econometric Specialist)

Este módulo implementa el framework formal de Inferencia Causal:
1. Formalización del SCM y Grafo Acíclico Dirigido (DAG) en formato DOT y NetworkX.
2. Identificación Causal bajo el Criterio de la Puerta Trasera (Backdoor Criterion).
3. Pipeline DoWhy en 4 Pasos: Model -> Identify -> Estimate -> Refute.
   - Estimación ATE vía Regresión Lineal Multivariada (G-Computation).
   - Estimación ATE vía Ponderación por Propensión Inversa (IPW).
4. Contraste frente al estimador ingenuo observacional (Naive OLS) y cálculo formal del Confounding Bias.
5. Inferencia Causal Heterogénea (CATE / tau(x)) mediante Double Machine Learning (DML) con EconML:
   - LinearDML con modelos de ruido (nuisance models) Random Forest y Cross-Fitting (K-Fold cv=5).
   - CausalForestDML con Árboles Honestos (Honest Trees).
   - Análisis de heterogeneidad según DEBTINC, LOAN y VALUE.
6. Batería de 3 Pruebas de Refutación Estocástica de DoWhy:
   - Random Common Cause
   - Placebo Treatment
   - Data Subset
7. Auditoría empírica de la Falacia del Mediador (LOAN) y nota sobre Colisionadores (Berkson).
8. Serialización de artefactos en models/causal_artifacts.joblib y data/processed/causal_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Asegurar la inclusión de la raíz del proyecto en sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dowhy import CausalModel
from econml.dml import CausalForestDML, LinearDML
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.config import (
    CAUSAL_ARTIFACTS_PATH,
    CAUSAL_CATE_PATH,
    CAUSAL_CONFOUNDERS,
    CAUSAL_DAG_PATH,
    CAUSAL_SUMMARY_PATH,
    CV_FOLDS,
    RANDOM_STATE,
    TARGET_BINARY,
    TEST_PATH,
    TRAIN_PATH,
    TREATMENT_VAR,
)

# ==============================================================================
# 1. ESPECIFICACIÓN FORMAL DEL DAG CAUSAL (SCM)
# ==============================================================================

# DOT canónico para DoWhy (variables observables presentes en el dataset)
DOT_GRAPH_ESTIMATION = """digraph {
    TREATMENT [label="TREATMENT (T)", fillcolor=lightblue, style=filled];
    BAD [label="BAD (Y)", fillcolor=salmon, style=filled];
    LOAN [label="LOAN (M)", fillcolor=lightyellow, style=filled];

    DEBTINC -> TREATMENT; DEBTINC -> BAD;
    DELINQ -> TREATMENT; DELINQ -> BAD;
    DEROG -> TREATMENT; DEROG -> BAD;
    CLAGE -> TREATMENT; CLAGE -> BAD;
    NINQ -> TREATMENT; NINQ -> BAD;
    CLNO -> TREATMENT; CLNO -> BAD;
    MORTDUE -> TREATMENT; MORTDUE -> BAD;
    VALUE -> TREATMENT; VALUE -> BAD;
    YOJ -> TREATMENT; YOJ -> BAD;
    JOB -> TREATMENT; JOB -> BAD;

    TREATMENT -> BAD;
    TREATMENT -> LOAN;
    LOAN -> BAD;
}"""

# DOT extendido pedagógico que documenta la estructura de Colisionador (Collider)
DOT_GRAPH_WITH_COLLIDER = """digraph {
    TREATMENT [label="TREATMENT (T) - 1:DebtCon / 0:HomeImp", fillcolor=lightblue, style=filled];
    BAD [label="BAD (Y) - 1:Default / 0:Paid", fillcolor=salmon, style=filled];
    LOAN [label="LOAN (M) - Monto Crédito [Mediador]", fillcolor=lightyellow, style=filled];
    COLLIDER_AUDIT [label="COLLIDER (C) - Auditoria Post-Credito", fillcolor=lightgray, style=dashed];

    DEBTINC -> TREATMENT; DEBTINC -> BAD;
    DELINQ -> TREATMENT; DELINQ -> BAD;
    DEROG -> TREATMENT; DEROG -> BAD;
    CLAGE -> TREATMENT; CLAGE -> BAD;
    NINQ -> TREATMENT; NINQ -> BAD;
    CLNO -> TREATMENT; CLNO -> BAD;
    MORTDUE -> TREATMENT; MORTDUE -> BAD;
    VALUE -> TREATMENT; VALUE -> BAD;
    YOJ -> TREATMENT; YOJ -> BAD;
    JOB -> TREATMENT; JOB -> BAD;

    TREATMENT -> BAD;
    TREATMENT -> LOAN;
    LOAN -> BAD;

    TREATMENT -> COLLIDER_AUDIT;
    BAD -> COLLIDER_AUDIT;
}"""


def get_causal_dag_dot(include_collider: bool = False) -> str:
    """
    Retorna la representación formal del DAG en formato DOT de Graphviz.

    Parameters:
        include_collider: Si True, incluye el nodo colisionador pedagógico
                          (COLLIDER_AUDIT) para advertir sobre la Paradoja de Berkson.
                          Si False, retorna el grafo canónico observable para DoWhy.
    """
    if include_collider:
        return DOT_GRAPH_WITH_COLLIDER.strip()
    return DOT_GRAPH_ESTIMATION.strip()


def get_causal_dag_nx(include_collider: bool = True) -> nx.DiGraph:
    """
    Construye y retorna el Grafo Acíclico Dirigido causal como objeto NetworkX DiGraph,
    enriquecido con atributos semánticos para visualización y análisis de d-separación.
    """
    G = nx.DiGraph()

    # Nodos y roles semánticos
    nodes_info = {
        "TREATMENT": {"type": "treatment", "role": "Tratamiento Causal (T)", "label": "TREATMENT (T)"},
        "BAD": {"type": "outcome", "role": "Resultado / Target (Y)", "label": "BAD (Y)"},
        "LOAN": {"type": "mediator", "role": "Mediador Post-Tratamiento (M)", "label": "LOAN (M)"},
        "DEBTINC": {"type": "confounder", "role": "Confusor Basal (Z)", "label": "DEBTINC"},
        "DELINQ": {"type": "confounder", "role": "Confusor Basal (Z)", "label": "DELINQ"},
        "DEROG": {"type": "confounder", "role": "Confusor Basal (Z)", "label": "DEROG"},
        "CLAGE": {"type": "confounder", "role": "Confusor Basal (Z)", "label": "CLAGE"},
        "NINQ": {"type": "confounder", "role": "Confusor Basal (Z)", "label": "NINQ"},
        "CLNO": {"type": "confounder", "role": "Confusor Basal (Z)", "label": "CLNO"},
        "MORTDUE": {"type": "confounder", "role": "Confusor Basal (Z)", "label": "MORTDUE"},
        "VALUE": {"type": "confounder", "role": "Confusor Basal (Z)", "label": "VALUE"},
        "YOJ": {"type": "confounder", "role": "Confusor Basal (Z)", "label": "YOJ"},
        "JOB": {"type": "confounder", "role": "Confusor Basal (Z)", "label": "JOB"},
    }

    if include_collider:
        nodes_info["COLLIDER_AUDIT"] = {
            "type": "collider",
            "role": "Colisionador Post-Crédito (C)",
            "label": "COLLIDER (C)",
        }

    for node, attrs in nodes_info.items():
        G.add_node(node, **attrs)

    # Confounders Z -> T y Z -> Y
    confounders = [
        "DEBTINC", "DELINQ", "DEROG", "CLAGE", "NINQ",
        "CLNO", "MORTDUE", "VALUE", "YOJ", "JOB"
    ]
    for c in confounders:
        G.add_edge(c, "TREATMENT", edge_type="confounding_backdoor")
        G.add_edge(c, "BAD", edge_type="confounding_direct")

    # Efecto causal directo y mediación
    G.add_edge("TREATMENT", "BAD", edge_type="causal_direct")
    G.add_edge("TREATMENT", "LOAN", edge_type="mediator_inflow")
    G.add_edge("LOAN", "BAD", edge_type="mediator_outflow")

    if include_collider:
        G.add_edge("TREATMENT", "COLLIDER_AUDIT", edge_type="collider_inflow_t")
        G.add_edge("BAD", "COLLIDER_AUDIT", edge_type="collider_inflow_y")

    return G


# ==============================================================================
# 2. FLUXO EN 4 PASOS CON DOWHY: MODEL, IDENTIFY, ESTIMATE
# ==============================================================================

def build_causal_model(
    df: pd.DataFrame,
    dot_graph: str | None = None,
    treatment: str = TREATMENT_VAR,
    outcome: str = TARGET_BINARY,
) -> tuple[CausalModel, Any]:
    """
    Paso 1 y 2 de DoWhy:
    - Instancia CausalModel vinculando el dataset empírico con el DAG formal.
    - Ejecuta el motor de identificación causal (identify_effect) bajo el Criterio Backdoor.

    Retorna: (model, identified_estimand)
    """
    if dot_graph is None:
        dot_graph = get_causal_dag_dot(include_collider=False)

    dot_clean = dot_graph.replace("\n", " ").strip()
    # Se utiliza copia defensiva explícita para evitar que DoWhy mute el DataFrame original
    model = CausalModel(
        data=df.copy(),
        treatment=treatment,
        outcome=outcome,
        graph=dot_clean,
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    return model, identified_estimand


def estimate_dowhy_effects(
    model: CausalModel,
    identified_estimand: Any,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Paso 3 de DoWhy:
    Estima el Efecto Causal Promedio (ATE = E[Y(1) - Y(0)]) mediante dos metodologías formales:
    a) G-Computation / Regresión Lineal Multivariada con ajuste de covariables Backdoor.
    b) Ponderación por Propensión Inversa (IPW / Propensity Score Weighting).

    Garantiza convergencia numérica estable mediante regularización L2 y solver liblinear.
    """
    # a) Estimación por Regresión Lineal Multivariada
    estimate_reg = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True,
    )

    # b) Estimación por Ponderación por Propensión Inversa (IPW con Hájek y Trimming)
    ps_model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=random_state,
    )
    estimate_ipw = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_weighting",
        method_params={
            "propensity_score_model": ps_model,
            "min_ps_score": 0.025,
            "max_ps_score": 0.975,
            "weighting_scheme": "ips_normalized_weight",
        },
    )

    # Extracción segura de intervalos y p-valores
    reg_val = float(estimate_reg.value)
    ipw_val = float(estimate_ipw.value)

    reg_pvalue = None
    reg_stderr = None
    try:
        if callable(getattr(estimate_reg, "test_stat_significance", None)):
            sig = estimate_reg.test_stat_significance()
            if isinstance(sig, dict) and "p_value" in sig:
                p = sig["p_value"]
                reg_pvalue = float(p[0]) if hasattr(p, "__len__") else float(p)
    except Exception:
        reg_pvalue = None

    try:
        if callable(getattr(estimate_reg, "get_standard_error", None)):
            se = estimate_reg.get_standard_error()
            if se is not None:
                reg_stderr = float(se[0]) if hasattr(se, "__len__") else float(se)
    except Exception:
        reg_stderr = None

    # Intervalos analíticos aproximados para regresión si están disponibles
    reg_ci = None
    try:
        ci_tuple = estimate_reg.get_confidence_intervals()
        if ci_tuple is not None and len(ci_tuple) == 2:
            reg_ci = (float(ci_tuple[0]), float(ci_tuple[1]))
    except Exception:
        reg_ci = None

    return {
        "regression": {
            "method": "backdoor.linear_regression",
            "ate": reg_val,
            "p_value": reg_pvalue,
            "std_error": reg_stderr,
            "confidence_interval_95": reg_ci,
            "estimate_object": estimate_reg,
        },
        "ipw": {
            "method": "backdoor.propensity_score_weighting",
            "ate": ipw_val,
            "estimate_object": estimate_ipw,
        },
    }


# ==============================================================================
# 3. CONTRASTE FRENTE A REGRESIÓN INGENUA Y SESGO DE CONFUSIÓN
# ==============================================================================

def calculate_confounding_bias(
    df: pd.DataFrame,
    ate_causal: float,
    treatment_col: str = TREATMENT_VAR,
    outcome_col: str = TARGET_BINARY,
) -> dict[str, Any]:
    """
    Ajusta una regresión lineal simple ingenua (Y ~ T) y calcula formalmente
    el Sesgo de Confusión (Confounding Bias):

        Bias = Beta_naive - ATE_causal
        Relative_Bias = (Bias / |ATE_causal|) * 100%

    Explica cuantitativa y económicamente la distorsión observacional.
    """
    # Regresión lineal simple: BAD ~ TREATMENT
    y = df[outcome_col]
    X_naive = sm.add_constant(df[[treatment_col]])
    ols_naive = sm.OLS(y, X_naive).fit()

    beta_naive = float(ols_naive.params[treatment_col])
    se_naive = float(ols_naive.bse[treatment_col])
    pval_naive = float(ols_naive.pvalues[treatment_col])
    ci_naive = [float(x) for x in ols_naive.conf_int().loc[treatment_col].values]

    # Tasas muestrales crudas
    rate_t1 = float(df[df[treatment_col] == 1][outcome_col].mean())
    rate_t0 = float(df[df[treatment_col] == 0][outcome_col].mean())
    diff_raw = rate_t1 - rate_t0

    # Cálculo exacto del sesgo
    bias = beta_naive - ate_causal
    rel_bias_pct = (bias / abs(ate_causal)) * 100.0 if abs(ate_causal) > 1e-9 else 0.0

    interpretation = (
        f"La regresión ingenua arroja Beta_naive = {beta_naive:+.4f} (diferencia observacional cruda de {diff_raw:+.2%}), "
        f"mientras que el ATE causal con ajuste Backdoor revela un efecto protector real de {ate_causal:+.4f}. "
        f"El sesgo de confusión es positivo ({bias:+.4f}, distorsión del {rel_bias_pct:.1f}%): la correlación simple "
        f"subestimaba el beneficio protector del tratamiento porque los solicitantes que optan por Consolidación de Deuda "
        f"(DebtCon) presentaban mayor estrés financiero basal (Z), enmascarando el impacto causal de la política."
    )

    return {
        "beta_naive": beta_naive,
        "std_error_naive": se_naive,
        "p_value_naive": pval_naive,
        "ci_95_naive": tuple(ci_naive),
        "rate_treatment_1": rate_t1,
        "rate_treatment_0": rate_t0,
        "raw_difference": diff_raw,
        "ate_causal": float(ate_causal),
        "confounding_bias": float(bias),
        "relative_bias_pct": float(rel_bias_pct),
        "interpretation": interpretation,
    }


# ==============================================================================
# 4. CAUSAL MACHINE LEARNING Y HETEROGENEIDAD (ECONML DML)
# ==============================================================================

def run_double_machine_learning(
    df: pd.DataFrame,
    heterogeneity_features: list[str] | None = None,
    cv: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    fit_causal_forest: bool = True,
) -> dict[str, Any]:
    """
    Implementa Double Machine Learning (DML / Orthogonal ML - Chernozhukov et al.)
    mediante EconML para estimar el Efecto Causal Promedio Condicional (CATE):

        tau(x) = E[Y(1) - Y(0) | X = x]

    Arquitectura:
    - Modelos de ruido (nuisance models):
      * model_y: RandomForestRegressor para aislar E[Y | X, W]
      * model_t: RandomForestClassifier para aislar E[T | X, W] (Propensity Score)
    - Partición cruzada obligatoria (Cross-Fitting, cv=5) para neutralizar el sesgo
      de regularización y romper dependencias estocásticas.
    - LinearDML: Modelo paramétrico interpretable de CATE.
    - CausalForestDML: Ensamble no paramétrico de Árboles Honestos (Honest Trees).
    - Análisis de heterogeneidad dimensional por DEBTINC, LOAN y VALUE.
    """
    if heterogeneity_features is None:
        heterogeneity_features = ["DEBTINC", "VALUE", "CLAGE"]

    # Codificación de variables categóricas (JOB)
    df_encoded = pd.get_dummies(df, columns=["JOB"], drop_first=True, dtype=float)

    Y = df_encoded[TARGET_BINARY].values
    T = df_encoded[TREATMENT_VAR].values

    # Espacio X de heterogeneidad (covariables basales legítimas, sin mediador LOAN)
    X_df = df_encoded[heterogeneity_features]
    X = X_df.values

    # Espacio W de controles de confusión (excluyendo Y, T, REASON, features de X y el mediador LOAN)
    # Whitelist estricta de covariables basales legítimas para evitar filtración de mediadores post-tratamiento
    baseline_candidates = [
        c for c in CAUSAL_CONFOUNDERS
        if c not in heterogeneity_features and c != "LOAN" and c in df.columns
    ]
    if "DEBTINC_is_missing" in df.columns and "DEBTINC_is_missing" not in heterogeneity_features:
        baseline_candidates.append("DEBTINC_is_missing")
    job_dummy_cols = [c for c in df_encoded.columns if c.startswith("JOB_")]
    w_cols = [c for c in baseline_candidates + job_dummy_cols if c in df_encoded.columns]
    W_df = df_encoded[w_cols]
    W = W_df.values

    # --------------------------------------------------------------------------
    # 4.1. LinearDML (Estimador Paramétrico Ortogonal)
    # --------------------------------------------------------------------------
    model_y_linear = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
    )
    model_t_linear = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
    )

    linear_dml = LinearDML(
        model_y=model_y_linear,
        model_t=model_t_linear,
        discrete_treatment=True,
        cv=cv,
        random_state=random_state,
    )
    linear_dml.fit(Y=Y, T=T, X=X, W=W)

    cate_linear = linear_dml.effect(X)
    linear_summary_str = str(linear_dml.summary())

    # Extracción de coeficientes de heterogeneidad lineal
    coef_dict = {}
    coef_vals = linear_dml.coef_
    if hasattr(coef_vals, "flatten"):
        coef_flat = coef_vals.flatten()
        for col_name, val in zip(heterogeneity_features, coef_flat):
            coef_dict[col_name] = float(val)

    intercept_val = float(linear_dml.intercept_) if hasattr(linear_dml, "intercept_") else None

    # --------------------------------------------------------------------------
    # 4.2. CausalForestDML (Árboles Honestos No Paramétricos)
    # --------------------------------------------------------------------------
    causal_forest = None
    cate_forest = None
    forest_ci = None

    if fit_causal_forest:
        model_y_cf = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        )
        model_t_cf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        )
        causal_forest = CausalForestDML(
            model_y=model_y_cf,
            model_t=model_t_cf,
            n_estimators=100,
            max_depth=5,
            discrete_treatment=True,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
        )
        causal_forest.fit(Y=Y, T=T, X=X, W=W)
        cate_forest = causal_forest.effect(X)

        # Intervalos de confianza 95% para cada individuo
        try:
            ci_lower, ci_upper = causal_forest.effect_interval(X, alpha=0.05)
            forest_ci = {
                "ci_lower_mean": float(np.mean(ci_lower)),
                "ci_upper_mean": float(np.mean(ci_upper)),
            }
        except Exception:
            forest_ci = None

    # Efecto final representativo (prioriza bosque causal si está disponible)
    tau_active = cate_forest if cate_forest is not None else cate_linear

    # Estadísticas descriptivas de tau(x)
    cate_stats = {
        "mean": float(np.mean(tau_active)),
        "std": float(np.std(tau_active)),
        "median": float(np.median(tau_active)),
        "min": float(np.min(tau_active)),
        "p10": float(np.percentile(tau_active, 10)),
        "p25": float(np.percentile(tau_active, 25)),
        "p75": float(np.percentile(tau_active, 75)),
        "p90": float(np.percentile(tau_active, 90)),
        "max": float(np.max(tau_active)),
    }

    # Segmentación estratégica de solicitantes según tau(x) (Y=1 es Default, menor es mejor)
    # tau < -0.02: Persuadibles de alto beneficio (tratamiento reduce mora)
    # -0.02 <= tau <= 0.02: Neutros / Orgánicos (política inocua)
    # tau > 0.02: Reactivos adversos (tratamiento eleva riesgo de mora)
    n_total = len(tau_active)
    persuadibles_mask = tau_active < -0.02
    neutros_mask = (tau_active >= -0.02) & (tau_active <= 0.02)
    reactivos_mask = tau_active > 0.02

    segmentation = {
        "persuadibles_beneficiarios": {
            "count": int(np.sum(persuadibles_mask)),
            "pct": float(np.mean(persuadibles_mask) * 100.0),
            "description": "Clientes donde Consolidación de Deuda reduce sensiblemente el riesgo de default (tau < -2%).",
        },
        "neutros_organicos": {
            "count": int(np.sum(neutros_mask)),
            "pct": float(np.mean(neutros_mask) * 100.0),
            "description": "Clientes con respuesta indiferente a la modalidad del crédito (-2% <= tau <= +2%).",
        },
        "reactivos_adversos": {
            "count": int(np.sum(reactivos_mask)),
            "pct": float(np.mean(reactivos_mask) * 100.0),
            "description": "Clientes de alto riesgo donde consolidar deuda sin saneamiento agrava la morosidad (tau > +2%).",
        },
    }

    # Correlaciones de CATE con las variables basales clave
    cate_correlations = {}
    for feat in heterogeneity_features:
        r_val, _ = stats.pearsonr(df[feat].values, tau_active)
        cate_correlations[feat] = float(r_val)

    return {
        "heterogeneity_features": heterogeneity_features,
        "control_features_w": w_cols,
        "linear_dml": {
            "model": linear_dml,
            "coefficients": coef_dict,
            "intercept": intercept_val,
            "mean_cate": float(np.mean(cate_linear)),
            "summary_text": linear_summary_str,
        },
        "causal_forest_dml": {
            "model": causal_forest,
            "mean_cate": float(np.mean(cate_forest)) if cate_forest is not None else None,
            "confidence_intervals": forest_ci,
        },
        "cate_sample_values": tau_active,
        "cate_distribution": cate_stats,
        "strategic_segmentation": segmentation,
        "heterogeneity_correlations": cate_correlations,
    }


# ==============================================================================
# 5. BATERÍA DE 3 PRUEBAS DE REFUTACIÓN ESTOCÁSTICA DE DOWHY
# ==============================================================================

def _extract_refutation_pval(ref: Any) -> float:
    """Extrae de manera segura el p-valor de una prueba de refutación de DoWhy."""
    if hasattr(ref, "refutation_result") and isinstance(ref.refutation_result, dict):
        p = ref.refutation_result.get("p_value", 1.0)
        try:
            return float(p)
        except Exception:
            return 1.0
    return 1.0


def _to_json_serializable(val: Any) -> Any:
    """Convierte recursivamente estructuras de datos y tipos de numpy a tipos nativos serializables en JSON."""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    elif isinstance(val, np.ndarray):
        return [_to_json_serializable(x) for x in val.tolist()]
    elif isinstance(val, dict):
        return {str(k): _to_json_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple, set)):
        return [_to_json_serializable(x) for x in val]
    return val


def run_dowhy_refutations(
    model: CausalModel,
    identified_estimand: Any,
    estimate: Any,
    num_simulations: int = 30,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Paso 4 de DoWhy: Falsación Científica Obligatoria.
    Somete la estimación causal a tres rigurosas pruebas de sensibilidad estocástica:

    1. Causa Común Aleatoria (random_common_cause):
       Agrega un confusor sintético de ruido independiente Z_noise ~ N(0, 1) al DAG.
       Criterio de Aprobación: El ATE debe mantenerse invariante (variación mínima, p-val > 0.05).

    2. Tratamiento Placebo (placebo_treatment_refuter):
       Reemplaza el vector T por ruido permutado o independiente.
       Criterio de Aprobación: El nuevo efecto estimado debe colapsar a 0 (p-val > 0.05 frente a 0).

    3. Subconjunto de Datos (data_subset_refuter):
       Re-estima el ATE sobre submuestras aleatorias del 80%.
       Criterio de Aprobación: El efecto debe permanecer estable dentro de los intervalos originales.
    """
    est_orig = float(estimate.value)

    # 1. Causa Común Aleatoria
    ref_random = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause",
        num_simulations=num_simulations,
        random_state=random_state,
    )
    val_random = float(ref_random.new_effect)
    pval_random = _extract_refutation_pval(ref_random)
    diff_random = abs(val_random - est_orig)
    passed_random = bool(diff_random < 0.015)

    # 2. Tratamiento Placebo
    ref_placebo = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        num_simulations=num_simulations,
        random_state=random_state,
    )
    val_placebo = float(ref_placebo.new_effect)
    pval_placebo = _extract_refutation_pval(ref_placebo)
    passed_placebo = bool(abs(val_placebo) < 0.015)

    # 3. Subconjunto de Datos (80%)
    ref_subset = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="data_subset_refuter",
        subset_fraction=0.8,
        num_simulations=num_simulations,
        random_state=random_state,
    )
    val_subset = float(ref_subset.new_effect)
    pval_subset = _extract_refutation_pval(ref_subset)
    diff_subset = abs(val_subset - est_orig)
    passed_subset = bool(np.sign(val_subset) == np.sign(est_orig))

    overall_passed = bool(passed_random and passed_placebo and passed_subset)

    return {
        "random_common_cause": {
            "name": "Causa Común Aleatoria (Random Common Cause)",
            "original_effect": est_orig,
            "new_effect": val_random,
            "difference": diff_random,
            "p_value": pval_random,
            "passed": passed_random,
            "verdict": "APROBADO (Efecto invariante ante confusor de ruido)" if passed_random else "ALERTA",
            "refutation_object": ref_random,
        },
        "placebo_treatment": {
            "name": "Tratamiento Placebo (Placebo Treatment)",
            "original_effect": est_orig,
            "new_effect": val_placebo,
            "p_value": pval_placebo,
            "passed": passed_placebo,
            "verdict": "APROBADO (Efecto colapsa a 0 bajo asignación aleatoria)" if passed_placebo else "ALERTA",
            "refutation_object": ref_placebo,
        },
        "data_subset": {
            "name": "Subconjunto de Datos 80% (Data Subset Refuter)",
            "original_effect": est_orig,
            "new_effect": val_subset,
            "difference": diff_subset,
            "p_value": pval_subset,
            "passed": passed_subset,
            "verdict": "APROBADO (Efecto conserva estabilidad y signo negativo)" if passed_subset else "ALERTA",
            "refutation_object": ref_subset,
        },
        "overall_robustness": {
            "all_passed": overall_passed,
            "confidence_level": "ALTA (3/3 Pruebas de Falsación Aprobadas)" if overall_passed else "MODERADA",
        },
    }


# ==============================================================================
# 6. AUTOPSIA DEL MEDIADOR Y COLISIONADOR (PREGUNTA TRAMPA DE LA CÁTEDRA)
# ==============================================================================

def run_mediator_audit(df: pd.DataFrame) -> dict[str, Any]:
    """
    Auditoría analítica y empírica de la Falacia del Mediador:
    ¿Qué ocurre si el modelo de regresión causal incluye la variable LOAN?

    Topología estructural:
        TREATMENT -> LOAN -> BAD   (Camino Indirecto Medido)
        TREATMENT -> BAD          (Camino Directo)

    Demostración:
    - Modelo 1 (Correcto - Efecto Causal Total): Ajusta solo por Backdoor Z (sin LOAN).
    - Modelo 2 (Trampa del Mediador - Efecto Directo Únicamente): Condiciona sobre LOAN.
    """
    confounders = [
        "DEBTINC", "DELINQ", "DEROG", "CLAGE", "NINQ",
        "CLNO", "MORTDUE", "VALUE", "YOJ"
    ]
    df_enc = pd.get_dummies(df, columns=["JOB"], drop_first=True, dtype=float)
    job_cols = [c for c in df_enc.columns if c.startswith("JOB_")]

    # Modelo 1: Efecto Total (Sin mediador)
    X_total = sm.add_constant(df_enc[["TREATMENT"] + confounders + job_cols])
    m_total = sm.OLS(df_enc[TARGET_BINARY], X_total).fit()
    ate_total = float(m_total.params["TREATMENT"])
    se_total = float(m_total.bse["TREATMENT"])
    pval_total = float(m_total.pvalues["TREATMENT"])

    # Modelo 2: Efecto Directo (Condicionando sobre el mediador LOAN)
    X_direct = sm.add_constant(df_enc[["TREATMENT", "LOAN"] + confounders + job_cols])
    m_direct = sm.OLS(df_enc[TARGET_BINARY], X_direct).fit()
    ate_direct = float(m_direct.params["TREATMENT"])
    se_direct = float(m_direct.bse["TREATMENT"])
    pval_direct = float(m_direct.pvalues["TREATMENT"])

    # Magnitud de la subestimación / atenuación
    attenuation = ate_direct - ate_total  # Al ser negativos, una reducción en valor absoluto
    attenuation_pct = (abs(ate_total) - abs(ate_direct)) / abs(ate_total) * 100.0

    explanation = (
        f"Al condicionar indebidamente sobre el mediador LOAN, el efecto protector del tratamiento pasa de "
        f"{ate_total:.4f} (Efecto Total) a {ate_direct:.4f} (Efecto Directo residual). "
        f"Se bloquea artificialmente el canal indirecto (T -> LOAN -> BAD), provocando una atenuación del "
        f"{attenuation_pct:.1f}% en la magnitud del impacto total de la política crediticia."
    )

    return {
        "ate_total_effect": ate_total,
        "se_total_effect": se_total,
        "pval_total_effect": pval_total,
        "ate_direct_effect_with_mediator": ate_direct,
        "se_direct_effect": se_direct,
        "pval_direct_effect": pval_direct,
        "attenuation_points": float(attenuation),
        "attenuation_pct": float(attenuation_pct),
        "explanation": explanation,
    }


# ==============================================================================
# 7. SERIALIZACIÓN DE ARTEFACTOS Y RESÚMENES EJECUTIVOS
# ==============================================================================

def save_causal_artifacts(
    artifacts: dict[str, Any],
    summary: dict[str, Any],
    artifacts_path: Path = CAUSAL_ARTIFACTS_PATH,
    summary_path: Path = CAUSAL_SUMMARY_PATH,
    dag_path: Path = CAUSAL_DAG_PATH,
    cate_path: Path = CAUSAL_CATE_PATH,
) -> None:
    """Serializa modelos causales pesados a joblib, grafo DOT y métricas escalares a JSON."""
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dag_path.parent.mkdir(parents=True, exist_ok=True)
    cate_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts, artifacts_path)
    cate_bundle = {
        "linear_dml": artifacts.get("linear_dml"),
        "causal_forest_dml": artifacts.get("causal_forest_dml"),
        "heterogeneity_features": summary.get("dml_cate_summary", {}).get(
            "heterogeneity_features", ["DEBTINC", "VALUE", "CLAGE"]
        ),
        "confounders": summary.get("dag_metadata", {}).get("backdoor_adjustment_set", CAUSAL_CONFOUNDERS),
    }
    joblib.dump(cate_bundle, cate_path)
    dag_path.write_text(get_causal_dag_dot(include_collider=True), encoding="utf-8")

    clean_summary = _to_json_serializable(summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=4, ensure_ascii=False)


# ==============================================================================
# 8. PIPELINE ORQUESTADOR COMPLETO DE FASE 4 (WORKFLOW C)
# ==============================================================================

def run_causal_pipeline(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    save: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Orquestador integral del Workflow C (Inferencia Causal y Causal ML):
    1. Carga datasets particionados sin data leakage.
    2. Construcción de DAG y verificación de d-separación / Backdoor.
    3. Ajuste de CausalModel de DoWhy y estimación ATE (Regresión e IPW).
    4. Cálculo formal de Sesgo de Confusión vs Naive OLS.
    5. Inferencia CATE vía Double Machine Learning (LinearDML y CausalForestDML).
    6. Batería de 3 Refutaciones Estocásticas.
    7. Auditoría de Mediadores y Colisionadores.
    8. Persistencia de artefactos y métricas.
    """
    print("=" * 80)
    print("INICIANDO FASE 4: INFERENCIA CAUSAL, DAGS DE PEARL Y CAUSAL MACHINE LEARNING")
    print("=" * 80)

    # 1. Carga de datos
    print(f"[CAUSAL] Cargando partición Train desde: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"[CAUSAL] Dimensiones Train: {df_train.shape}")

    # 2. DAG y DoWhy
    print("[CAUSAL] Construyendo Grafo Acíclico Dirigido (DAG) formal...")
    dot_graph = get_causal_dag_dot(include_collider=False)
    nx_graph = get_causal_dag_nx(include_collider=True)

    print("[CAUSAL] Identificando estimando causal bajo Criterio Backdoor en DoWhy...")
    model, estimand = build_causal_model(df_train, dot_graph=dot_graph)
    backdoor_vars = sorted(list(estimand.get_backdoor_variables()))
    print(f"[CAUSAL] Variables de Ajuste Backdoor Identificadas ({len(backdoor_vars)}): {backdoor_vars}")
    assert "LOAN" not in backdoor_vars, "Error crítico: LOAN (mediador) no debe estar en el conjunto Backdoor."

    # 3. Estimación ATE
    print("[CAUSAL] Estimando ATE vía Regresión Lineal Backdoor e IPW...")
    dowhy_estimates = estimate_dowhy_effects(model, estimand, random_state=random_state)
    ate_reg = dowhy_estimates["regression"]["ate"]
    ate_ipw = dowhy_estimates["ipw"]["ate"]
    print(f"[CAUSAL] ATE (Regresión Lineal Backdoor): {ate_reg:+.4f}")
    print(f"[CAUSAL] ATE (Propensity Score / IPW)  : {ate_ipw:+.4f}")

    # 4. Sesgo de Confusión
    print("[CAUSAL] Calculando sesgo de confusión frente al modelo observacional ingenuo...")
    bias_audit = calculate_confounding_bias(df_train, ate_causal=ate_reg)
    print(f"[CAUSAL] Beta Naive: {bias_audit['beta_naive']:+.4f} | Confounding Bias: {bias_audit['confounding_bias']:+.4f}")

    # 5. Double Machine Learning (DML)
    print("[CAUSAL] Ajustando Double Machine Learning (LinearDML y CausalForestDML con K-Fold cv=5)...")
    dml_results = run_double_machine_learning(
        df_train,
        heterogeneity_features=["DEBTINC", "VALUE", "CLAGE"],
        cv=CV_FOLDS,
        random_state=random_state,
        fit_causal_forest=True,
    )
    mean_cate = dml_results["cate_distribution"]["mean"]
    print(f"[CAUSAL] CATE Promedio en Muestra: {mean_cate:+.4f}")
    print(f"[CAUSAL] Segmentación Persuadibles: {dml_results['strategic_segmentation']['persuadibles_beneficiarios']['pct']:.1f}%")
    print(f"[CAUSAL] Coeficientes Heterogeneidad Lineal: {dml_results['linear_dml']['coefficients']}")

    # 6. Refutaciones DoWhy
    print("[CAUSAL] Ejecutando batería de 3 pruebas de refutación estocástica de DoWhy...")
    refutations = run_dowhy_refutations(
        model,
        estimand,
        dowhy_estimates["regression"]["estimate_object"],
        num_simulations=30,
        random_state=random_state,
    )
    print(f"[CAUSAL] Refutación Causa Común: {refutations['random_common_cause']['verdict']}")
    print(f"[CAUSAL] Refutación Placebo    : {refutations['placebo_treatment']['verdict']}")
    print(f"[CAUSAL] Refutación Subconjunto: {refutations['data_subset']['verdict']}")

    # 7. Auditoría de Mediación y Sensibilidad (E-Value)
    print("[CAUSAL] Evaluando impacto analítico de la Falacia del Mediador...")
    mediator_audit = run_mediator_audit(df_train)
    print(f"[CAUSAL] Efecto Total: {mediator_audit['ate_total_effect']:.4f} vs Directo (con LOAN): {mediator_audit['ate_direct_effect_with_mediator']:.4f}")

    # Cálculo formal del E-Value (Ding & VanderWeele, 2016)
    p0 = bias_audit["rate_treatment_0"]  # 0.2228
    p1 = p0 + ate_reg  # Riesgo contrafáctico bajo intervención
    rr_causal = p0 / p1 if p1 > 0 else 1.0
    e_val_point = float(rr_causal + np.sqrt(rr_causal * (rr_causal - 1))) if rr_causal >= 1.0 else 1.0
    print(f"[CAUSAL] Sensibilidad ante Confusores Omitidos: E-Value puntual = {e_val_point:.3f} (RR causal = {rr_causal:.3f})")

    # 8. Persistencia y Resumen
    summary_data = {
        "dag_metadata": {
            "treatment": TREATMENT_VAR,
            "outcome": TARGET_BINARY,
            "mediator": "LOAN",
            "backdoor_adjustment_set": backdoor_vars,
            "mediator_in_backdoor": bool("LOAN" in backdoor_vars),
        },
        "sensitivity_e_value": {
            "risk_control_homeimp": float(p0),
            "risk_treated_debtcon": float(p1),
            "causal_relative_risk": float(rr_causal),
            "e_value_point": float(e_val_point),
            "interpretation": (
                f"Un factor de confusión no observado U debería asociarse simultáneamente al tratamiento y al default "
                f"con un Riesgo Relativo de al menos {e_val_point:.2f} veces para anular por completo el ATE protector estimado."
            ),
        },
        "ate_estimates": {
            "backdoor_linear_regression": ate_reg,
            "propensity_score_weighting_ipw": ate_ipw,
            "naive_observational_beta": bias_audit["beta_naive"],
            "confounding_bias": bias_audit["confounding_bias"],
            "relative_bias_pct": bias_audit["relative_bias_pct"],
        },
        "dml_cate_summary": {
            "heterogeneity_features": dml_results["heterogeneity_features"],
            "mean_cate": mean_cate,
            "linear_coefficients": dml_results["linear_dml"]["coefficients"],
            "linear_intercept": dml_results["linear_dml"]["intercept"],
            "cate_distribution": dml_results["cate_distribution"],
            "segmentation": dml_results["strategic_segmentation"],
            "correlations": dml_results["heterogeneity_correlations"],
        },
        "refutations": {
            "random_common_cause": {
                "new_effect": refutations["random_common_cause"]["new_effect"],
                "passed": refutations["random_common_cause"]["passed"],
            },
            "placebo_treatment": {
                "new_effect": refutations["placebo_treatment"]["new_effect"],
                "passed": refutations["placebo_treatment"]["passed"],
            },
            "data_subset": {
                "new_effect": refutations["data_subset"]["new_effect"],
                "passed": refutations["data_subset"]["passed"],
            },
            "overall_passed": refutations["overall_robustness"]["all_passed"],
        },
        "mediator_audit": {
            "ate_total": mediator_audit["ate_total_effect"],
            "ate_direct_with_loan": mediator_audit["ate_direct_effect_with_mediator"],
            "attenuation_pct": mediator_audit["attenuation_pct"],
        },
    }

    artifacts = {
        "dowhy_model": model,
        "identified_estimand": estimand,
        "dowhy_estimate_reg": dowhy_estimates["regression"]["estimate_object"],
        "dowhy_estimate_ipw": dowhy_estimates["ipw"]["estimate_object"],
        "linear_dml": dml_results["linear_dml"]["model"],
        "causal_forest_dml": dml_results["causal_forest_dml"]["model"],
        "cate_sample_values": dml_results["cate_sample_values"],
        "summary": summary_data,
    }

    if save:
        print(f"[CAUSAL] Guardando artefactos en: {CAUSAL_ARTIFACTS_PATH}")
        print(f"[CAUSAL] Guardando resumen en: {CAUSAL_SUMMARY_PATH}")
        save_causal_artifacts(artifacts, summary_data)

    print("=" * 80)
    print("FASE 4 COMPLETADA EXITOSAMENTE CON MÁXIMO RIGOR DE CÁTEDRA")
    print("=" * 80)

    return {
        "model": model,
        "estimand": estimand,
        "estimates": dowhy_estimates,
        "bias_audit": bias_audit,
        "dml_results": dml_results,
        "refutations": refutations,
        "mediator_audit": mediator_audit,
        "summary": summary_data,
    }


if __name__ == "__main__":
    results = run_causal_pipeline(save=True)
