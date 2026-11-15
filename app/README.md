# Fase 5 — Web App interactiva

App de Streamlit con tres pestañas: **Diagnóstico** (filtros, variables y PCA 2D),
**What-Is** (predicción del Random Forest para un solicitante) y **What-If**
(DAG, ATE, CATE y semáforo de refutaciones).

La app solo lee los artefactos de `artifacts/`. No instala DoWhy ni EconML: el CATE
del bosque causal se precalcula una vez con `python -m src.export_app_cate`, que
genera `artifacts/app_cate_grid.npz`.

## Correrla en la computadora (plan B para la presentación)

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

Se abre sola en `http://localhost:8501`. Hay que ejecutarla desde la raíz del repo.

## Publicarla en Streamlit Community Cloud

1. Entrar a https://share.streamlit.io con la cuenta de GitHub dueña del repo.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository: `gsittner/TPI-EayMR` · Branch: `main` · Main file path: `app/app.py`.
4. En **Advanced settings**, elegir **Python 3.13**.
5. **Deploy**. La primera vez tarda unos minutos en instalar las librerías.

Si nadie la usa por un tiempo, la app se duerme: abrir el link unos minutos antes
de presentar y tocar el botón para despertarla.

## Si se vuelve a correr la Fase 4

Regenerar el CATE de la app para que coincida con el nuevo modelo:

```bash
python -m src.export_app_cate
```
