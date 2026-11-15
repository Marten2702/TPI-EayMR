"""
src/export_app_cate.py
Precalcula el CATE del CausalForestDML (Fase 4) para la Web App (Fase 5).

La app en la nube no instala DoWhy ni EconML. Por eso este script, que se corre una
sola vez en una máquina con EconML, evalúa el bosque causal ya entrenado
(artifacts/causal_cate_model.joblib) y guarda los resultados en un archivo liviano:

  - una grilla regular sobre DEBTINC x VALUE x CLAGE con tau(x) y su IC 95 %,
    que la app interpola para cualquier perfil;
  - tau(x) y su IC 95 % para cada cliente de train, para simular políticas por segmento.

Uso:  python -m src.export_app_cate
"""
import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, CAUSAL_CATE_PATH, TRAIN_PATH

APP_CATE_PATH = ARTIFACTS_DIR / "app_cate_grid.npz"
PUNTOS_POR_EJE = 25


def exportar():
    bundle = joblib.load(CAUSAL_CATE_PATH)
    bosque, variables = bundle["causal_forest_dml"], bundle["heterogeneity_features"]
    train = pd.read_csv(TRAIN_PATH)

    # Ejes entre los percentiles 1 y 99 de train: fuera de ahí casi no hay datos
    ejes = [np.linspace(*np.percentile(train[v], [1, 99]), PUNTOS_POR_EJE) for v in variables]
    malla = np.stack(np.meshgrid(*ejes, indexing="ij"), axis=-1).reshape(-1, len(variables))
    forma = (PUNTOS_POR_EJE,) * len(variables)
    inf, sup = bosque.effect_interval(malla, alpha=0.05)

    X = train[variables].values
    inf_tr, sup_tr = bosque.effect_interval(X, alpha=0.05)

    np.savez_compressed(
        APP_CATE_PATH,
        variables=np.array(variables),
        **{f"eje_{i}": e for i, e in enumerate(ejes)},
        tau=bosque.effect(malla).reshape(forma).astype("float32"),
        tau_inf=inf.reshape(forma).astype("float32"),
        tau_sup=sup.reshape(forma).astype("float32"),
        tau_train=bosque.effect(X).astype("float32"),
        tau_train_inf=inf_tr.astype("float32"),
        tau_train_sup=sup_tr.astype("float32"),
    )
    print(f"[APP] CATE precalculado en: {APP_CATE_PATH}")


if __name__ == "__main__":
    exportar()
