"""
tests/test_artifacts.py
Integridad de los archivos de artifacts/ que consume la Web App (Fase 5).
"""

import json

import numpy as np

from src.config import (ARTIFACTS_DIR, BEST_CLASSIFIER_PATH, CAUSAL_CATE_PATH, CAUSAL_DAG_PATH,
                        CAUSAL_SUMMARY_PATH, ML_METRICS_PATH, PCA_PIPELINE_PATH)

APP_CATE_PATH = ARTIFACTS_DIR / "app_cate_grid.npz"


def test_archivos_presentes():
    for ruta in (PCA_PIPELINE_PATH, BEST_CLASSIFIER_PATH, ML_METRICS_PATH, CAUSAL_DAG_PATH,
                 CAUSAL_CATE_PATH, CAUSAL_SUMMARY_PATH, APP_CATE_PATH):
        assert ruta.exists(), f"Falta {ruta.name}"


def test_metricas_del_ganador():
    m = json.loads(ML_METRICS_PATH.read_text(encoding="utf-8"))
    g = m["winner"]
    assert g["model_name"] == "Random Forest"
    assert 0.9 < g["test_roc_auc"] <= 1 and 0 < g["test_brier"] < 0.1
    assert {"n_estimators", "max_depth", "max_features"} <= set(g["best_params"])


def test_dag_exportado():
    texto = CAUSAL_DAG_PATH.read_text(encoding="utf-8")
    assert "TREATMENT -> BAD" in texto.replace('"', "")


def test_grilla_cate_de_la_app():
    z = np.load(APP_CATE_PATH)
    variables = [str(v) for v in z["variables"]]
    assert len(variables) == 3
    forma = tuple(len(z[f"eje_{i}"]) for i in range(3))
    assert z["tau"].shape == forma
    assert np.all(z["tau_inf"] <= z["tau_sup"])
    assert len(z["tau_train"]) == 4768
