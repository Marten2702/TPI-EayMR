"""
tests/test_phase3.py
Tests de la Fase 3 (Torneo predictivo y calibración).
Usan la partición congelada del Sprint 0 y modelos chicos para correr rápido.
"""

import pytest
from sklearn.metrics import roc_auc_score

from src.config import TARGET_BINARY
from src.phase3_ml import (
    fit_decision_tree_pruning,
    fit_naive_bayes_and_calibration,
    load_phase3_data,
)


@pytest.fixture(scope="module")
def datos():
    return load_phase3_data()


def test_matrices_sin_fuga(datos):
    """Train y test comparten columnas, no incluyen el objetivo ni REASON y no tienen faltantes."""
    _, _, X_train, X_test, y_train, y_test = datos
    assert X_train.shape == (4768, 17) and X_test.shape == (1192, 17)
    assert list(X_train.columns) == list(X_test.columns)
    assert TARGET_BINARY not in X_train.columns and "REASON" not in X_train.columns
    assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0
    assert abs(y_train.mean() - y_test.mean()) < 0.01  # partición estratificada


def test_poda_reduce_sobreajuste(datos):
    """El árbol podado es más chico y generaliza mejor que el árbol sin poda."""
    _, _, X_train, X_test, y_train, y_test = datos
    res = fit_decision_tree_pruning(X_train, y_train, X_test, y_test)
    completo, podado = res["tree_full"], res["tree_pruned"]
    auc = lambda m: roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    assert podado.get_n_leaves() < completo.get_n_leaves()
    assert auc(podado) > auc(completo)
    assert roc_auc_score(y_train, completo.predict_proba(X_train)[:, 1]) == pytest.approx(1.0)


def test_calibracion_naive_bayes(datos):
    """La calibración isotónica acerca la probabilidad media a la tasa real sin cambiar el orden."""
    _, _, X_train, X_test, y_train, y_test = datos
    res = fit_naive_bayes_and_calibration(X_train, y_train, X_test, y_test)
    p_raw = res["nb_raw"].predict_proba(X_test)[:, 1]
    p_cal = res["nb_calibrated"].predict_proba(X_test)[:, 1]
    tasa = y_test.mean()
    assert abs(p_cal.mean() - tasa) < abs(p_raw.mean() - tasa)
    assert p_raw.mean() < tasa  # Naive Bayes original subestima el riesgo
