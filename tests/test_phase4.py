"""
tests/test_phase4.py
Tests de la Fase 4 (Inferencia causal): estructura del DAG, criterio backdoor
y coherencia de los resultados guardados en artifacts/causal_summary.json.
"""

import json

import networkx as nx

from src.config import CAUSAL_CONFOUNDERS, CAUSAL_MEDIATOR, CAUSAL_SUMMARY_PATH
from src.phase4_causal import get_causal_dag_nx


def test_dag_aciclico_y_flecha_causal():
    """El grafo es acíclico y contiene la flecha TREATMENT -> BAD."""
    G = get_causal_dag_nx(include_collider=True)
    assert nx.is_directed_acyclic_graph(G)
    assert G.has_edge("TREATMENT", "BAD")


def test_mediador_fuera_del_ajuste():
    """LOAN es descendiente del tratamiento y no forma parte del conjunto de ajuste."""
    G = get_causal_dag_nx(include_collider=False)
    assert CAUSAL_MEDIATOR == "LOAN"
    assert CAUSAL_MEDIATOR not in CAUSAL_CONFOUNDERS
    assert CAUSAL_MEDIATOR in nx.descendants(G, "TREATMENT")
    resumen = json.loads(CAUSAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert CAUSAL_MEDIATOR not in resumen["dag_metadata"]["backdoor_adjustment_set"]


def test_confusores_abren_puertas_traseras():
    """Cada confusor apunta al tratamiento y al resultado, y ninguno desciende del tratamiento."""
    G = get_causal_dag_nx(include_collider=False)
    descendientes = nx.descendants(G, "TREATMENT")
    for z in CAUSAL_CONFOUNDERS:
        assert G.has_edge(z, "TREATMENT") and G.has_edge(z, "BAD")
        assert z not in descendientes


def test_resultados_causales_coherentes():
    """El ATE ajustado es negativo, el placebo cae a cero y las tres refutaciones pasan."""
    resumen = json.loads(CAUSAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    ate = resumen["ate_estimates"]["backdoor_linear_regression"]
    ref = resumen["refutations"]
    assert ate < 0
    assert abs(ref["placebo_treatment"]["new_effect"]) < 0.01
    assert abs(ref["random_common_cause"]["new_effect"] - ate) < 0.005
    assert ref["data_subset"]["new_effect"] < 0
    assert ref["overall_passed"]
