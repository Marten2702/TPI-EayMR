# Contrato de Artefactos para Integración en Fase 5 (Web App)

Este directorio contiene los modelos, escaladores, grafos y métricas serializadas de cada fase.  
Todas las rutas están centralizadas en [`src/config.py`](../src/config.py).

---

## ✅ Fases 1 y 2 (Ya Exportadas)
Generado por: `python -m src.export_phase1_2`
1. `artifacts/pca_pipeline.joblib`
   - Diccionario con `feature_cols`, `scaler` (`StandardScaler` ajustado en Train), `pca` (`PCA` ajustado en Train), `eigenvalues`, `var_exp_ratio`, `loadings`, `X_train_pca_2d`, `X_test_pca_2d`.
2. `artifacts/phase1_2_summary.json`
   - Resumen completo de: mecanismo de pérdida MAR/MNAR, descriptivos EDA, contrastes 1 y 2, diagnóstico ANOVA / Levene / Welch / Kruskal-Wallis / Bonferroni, MCO, Gauss-Markov, Cook's $D$, regresores robustos y VIF.

---

## ✅ Fase 3 — Torneo Predictivo de ML (Martín Peña — Ya Exportada)
Módulo en `src/phase3_ml.py` (`python -m src.phase3_ml`):
1. `artifacts/best_classifier.joblib`
   - `Pipeline` (`ColumnTransformer` + `RandomForestClassifier`, 500 árboles) capaz de recibir un `pd.DataFrame` con las columnas de `train.csv` (o perfil individual de la Web App) y ejecutar `.predict_proba(df_input)[:, 1]`.
2. `artifacts/ml_metrics.json`
   - Incluye:
     - Tabla comparativa en Test de las 4 familias (Logística Elastic Net, Naive Bayes original y calibrado isotónico, CART podado `ccp_alpha`, Random Forest optimizado): `AUC_ROC`, `PR_AUC`, `F1_macro`, `F1_weighted`, `Precision`, `Recall`, `Brier_Score`.
     - Puntos de las **Curvas ROC, Precision-Recall y Calibración (Reliability Diagrams)**.
     - Ranking de importancia de variables: **MDI (Gini)** vs. **Permutation Importance en Test** + tabla de ablación `DEBTINC_is_missing`.

---

## ✅ Fase 4 — Inferencia Causal y Causal ML (Santiago Salvatore — Ya Exportada)
Módulo en `src/phase4_causal.py` (`python -m src.phase4_causal`):
1. `artifacts/causal_dag.dot`
   - Grafo Acíclico Dirigido en formato DOT definiendo $T$ (`TREATMENT`), $Y$ (`BAD`), 10 confusores $Z$, mediador `LOAN` ($M$) y colisionador `COLLIDER_AUDIT` ($C$).
2. `artifacts/causal_cate_model.joblib`
   - Bundle con `linear_dml`, `causal_forest_dml`, `heterogeneity_features` (`["DEBTINC", "VALUE", "CLAGE"]`) y `confounders` para estimar $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$ vía `.effect(X)`.
3. `artifacts/causal_artifacts.joblib`
   - Artefactos completos `DoWhy` (`dowhy_model`, `identified_estimand`, estimadores Regresión e IPW) y `EconML` (`LinearDML`, `CausalForestDML`).
4. `artifacts/causal_summary.json`
   - Incluye `dag_metadata`, `sensitivity_e_value`, `ate_estimates`, `dml_cate_summary`, `refutations` (3/3 para el semáforo de la Web App) y `mediator_audit`.
