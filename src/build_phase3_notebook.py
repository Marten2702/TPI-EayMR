"""
src/build_phase3_notebook.py
Generador modular de celdas para la FASE 3: Torneo de Modelos Predictivos y Calibración.
Cátedra: Estadística Aplicada y Modelos de Regresión (EAyMR 2026) — UCA
Responsable de Fase: Martín Pena (Machine Learning Engineer)
"""

import nbformat as nbf


def get_phase3_cells() -> list:
    """Retorna la lista ordenada de celdas Markdown y Code de Fase 3 para TP_Integrador_Final.ipynb."""
    cells = []

    cells.append(nbf.v4.new_markdown_cell(r"""---

# 🌲 FASE 3: Torneo de Modelos Predictivos y Calibración *(Unidades 4 y 5)*

**Responsable de Workflow:** Martín Pena (Machine Learning Engineer)  
**Pregunta de la fase:** Dado el perfil de un solicitante, ¿qué probabilidad calibrada tiene de entrar en default (`BAD = 1`)?  
Es una pregunta estrictamente **observacional / predictiva** (Nivel 1 de Pearl: $P(Y \mid X)$). Evaluamos cuatro familias sobre la partición congelada (`train.csv` = 4.768, `test.csv` = 1.192) sin fuga de datos.

**Pila y Protocolo:**
1. Regresión Logística Clásica (`statsmodels.Logit` con Odds Ratios) + Regularización Elastic Net (`GridSearchCV`, `cv=5`).
2. Naive Bayes (`GaussianNB`) + Auditoría de correlación intra-clase + **Calibración Isotónica**.
3. Árbol de Decisión CART completo vs. Poda Cost-Complexity (`ccp_alpha` óptimo en CV).
4. Random Forest (`RandomizedSearchCV` + convergencia *Out-Of-Bag*).
5. Comparación unificada en Test (`Precision`, `Recall`, `F1`, `AUC-ROC`, `PR-AUC`, `Brier Score`), Diagramas de Calibración, MDI vs. Permutation Importance y ablación de `DEBTINC_is_missing`.
"""))

    cells.append(nbf.v4.new_code_cell(r"""from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.calibration import calibration_curve

from src.phase3_ml import (
    load_phase3_data,
    fit_logistic_suite,
    fit_naive_bayes_and_calibration,
    fit_decision_tree_pruning,
    fit_random_forest_tournament,
    evaluate_models_test,
    compute_feature_importance_comparison,
    run_debtinc_missing_ablation,
)

train_ml, test_ml, X_train_ml, X_test_ml, y_train_ml, y_test_ml = load_phase3_data()
print(f"Matriz X_train: {X_train_ml.shape} | Tasa default Train: {y_train_ml.mean():.2%}")
print(f"Matriz X_test : {X_test_ml.shape} | Tasa default Test : {y_test_ml.mean():.2%}")
print(f"17 predictoras (referencia JOB = Mgr): {list(X_train_ml.columns)}")
"""))

    # 3.1 Logit y ElasticNet
    cells.append(nbf.v4.new_markdown_cell(r"""### 3.1 Regresión Logística Clásica (Odds Ratios) y Regularización Elastic Net
Ajustamos primero `Logit` sin penalización para interpretar Odds Ratios *ceteris paribus*, y luego evaluamos `GridSearchCV` sobre Elastic Net (`Ridge L2`, `Elastic Net`, `Lasso L1`).
"""))

    cells.append(nbf.v4.new_code_cell(r"""logit_res = fit_logistic_suite(X_train_ml, y_train_ml)
print(f"Pseudo R² de McFadden: {logit_res['pseudo_r2']:.4f} | Test LR global: p = {logit_res['llr_pvalue']:.2e}")
display(logit_res['or_table'].round(4))

print(f"\nMejor configuración Elastic Net: {logit_res['best_params']} -> AUC CV = {logit_res['best_cv_auc']:.4f}")
cv_grid = logit_res['cv_results'].pivot(
    index="param_logisticregression__C",
    columns="param_logisticregression__l1_ratio",
    values="mean_test_score",
).rename_axis(index="C", columns="l1_ratio")
display(cv_grid.round(4))

print("Comparación de coeficientes estandarizados (C=0.01): Ridge vs. Lasso")
display(logit_res['pen_comparison'].sort_values("Ridge_L2", key=abs, ascending=False).round(3))
"""))

    # 3.2 Naive Bayes y Calibración
    cells.append(nbf.v4.new_markdown_cell(r"""### 3.2 Naive Bayes (`GaussianNB`) y Calibración Isotónica
`GaussianNB` asume independencia condicional entre features dado `BAD`. Auditamos la correlación intra-clase real y corregimos la subestimación probabilística mediante **Calibración Isotónica (`cv=5`)**.
"""))

    cells.append(nbf.v4.new_code_cell(r"""nb_res = fit_naive_bayes_and_calibration(X_train_ml, y_train_ml, X_test_ml, y_test_ml)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, corr_df, title in zip(
    axes,
    [nb_res['corr_paid'], nb_res['corr_default']],
    ["Correlación intra-clase: Cumplidores (BAD = 0)", "Correlación intra-clase: Default (BAD = 1)"]
):
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                cbar=False, square=True, annot_kws={"size": 7}, ax=ax)
    ax.set_title(title, fontweight="bold")
plt.tight_layout()
plt.show()

display(nb_res['calibration_comparison'].round(4))
"""))

    # 3.3 CART y Poda
    cells.append(nbf.v4.new_markdown_cell(r"""### 3.3 Árbol de Decisión CART: Sobreajuste sin Poda vs. Poda `ccp_alpha`
Un árbol sin restricciones memoriza `train.csv`. Aplicamos **Cost-Complexity Pruning** eligiendo el $\alpha$ óptimo por validación cruzada estratificada.
"""))

    cells.append(nbf.v4.new_code_cell(r"""tree_res = fit_decision_tree_pruning(X_train_ml, y_train_ml, X_test_ml, y_test_ml)
print(f"Árbol sin poda : {tree_res['full_stats']['leaves']} hojas (prof={tree_res['full_stats']['depth']}) | AUC Train={tree_res['full_stats']['auc_train']:.4f} | AUC Test={tree_res['full_stats']['auc_test']:.4f}")
print(f"Árbol podado   : {tree_res['pruned_stats']['leaves']} hojas (prof={tree_res['pruned_stats']['depth']}) | AUC Train={tree_res['pruned_stats']['auc_train']:.4f} | AUC Test={tree_res['pruned_stats']['auc_test']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
curve = tree_res['pruning_curve']
axes[0].plot(curve['alpha'], curve['auc_cv'], 'o-')
axes[0].axvline(tree_res['best_alpha'], color='red', ls='--', label=f"α óptimo = {tree_res['best_alpha']:.5f}")
axes[0].set_xscale('log')
axes[0].set(title='Curva de Poda Cost-Complexity (CV AUC)', xlabel='ccp_alpha (log)', ylabel='AUC CV')
axes[0].legend()

plot_tree(tree_res['tree_pruned'], max_depth=2, feature_names=list(X_train_ml.columns),
          class_names=["Cumple", "Default"], filled=True, rounded=True, proportion=True, fontsize=9, ax=axes[1])
axes[1].set_title('Primeros niveles del Árbol CART Podado', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    # 3.4 Random Forest + OOB
    cells.append(nbf.v4.new_markdown_cell(r"""### 3.4 Random Forest (`RandomizedSearchCV`) y Convergencia *Out-Of-Bag* (OOB)
Ensamblamos árboles decorrelacionados por *bagging* y submuestreo de variables (`max_features`), validando convergencia mediante la cota OOB sin consumo de partición Test.
"""))

    cells.append(nbf.v4.new_code_cell(r"""rf_res = fit_random_forest_tournament(X_train_ml, y_train_ml)
print(f"Mejores hiperparámetros RF: {rf_res['best_params']}")
print(f"AUC ROC en CV (Train)     : {rf_res['best_cv_auc']:.4f}")

oob = rf_res['oob_curve']
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].plot(oob['n_arboles'], oob['error_oob'], 'o-')
axes[0].set(title='Convergencia del Error OOB', xlabel='Número de Árboles', ylabel='Error OOB')
axes[1].plot(oob['n_arboles'], oob['auc_oob'], 'o-', label='AUC OOB')
axes[1].axhline(rf_res['best_cv_auc'], color='red', ls='--', label=f"AUC CV = {rf_res['best_cv_auc']:.4f}")
axes[1].set(title='AUC OOB vs. Validación Cruzada', xlabel='Número de Árboles', ylabel='AUC-ROC')
axes[1].legend()
plt.tight_layout()
plt.show()
display(oob.round(4))
"""))

    # 3.5 Evaluación en Test y Reliability Diagrams
    cells.append(nbf.v4.new_markdown_cell(r"""### 3.5 Evaluación Final en Test: Torneo Comparativo, Curvas ROC/PR y Calibración
Comparamos las 4 familias sobre los 1.192 casos ciegos de `test.csv`.
"""))

    cells.append(nbf.v4.new_code_cell(r"""models_ml = {
    "Logística (Elastic Net)": logit_res['best_model'],
    "Naive Bayes": nb_res['nb_raw'],
    "Naive Bayes Calibrado": nb_res['nb_calibrated'],
    "Árbol sin poda": tree_res['tree_full'],
    "Árbol podado": tree_res['tree_pruned'],
    "Random Forest": rf_res['best_model'],
}
eval_res = evaluate_models_test(models_ml, X_test_ml, y_test_ml)
display(eval_res['metrics_table'].round(4))

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
for nombre in ["Logística (Elastic Net)", "Naive Bayes", "Naive Bayes Calibrado", "Árbol podado", "Random Forest"]:
    m_row = eval_res['metrics_table'].loc[nombre]
    roc_d = eval_res['roc_curves'][nombre]
    pr_d = eval_res['pr_curves'][nombre]
    cal_d = eval_res['calibration_curves'][nombre]
    axes[0].plot(roc_d['fpr'], roc_d['tpr'], lw=2, label=f"{nombre} ({m_row['AUC_ROC']:.3f})")
    axes[1].plot(pr_d['recall'], pr_d['precision'], lw=2, label=f"{nombre} ({m_row['PR_AUC']:.3f})")
    axes[2].plot(cal_d['pred'], cal_d['obs'], 'o-', lw=1.8, label=f"{nombre} (ECE={cal_d['ece_quantile']:.3f})")

axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set(title='Curvas ROC (Test)', xlabel='FPR', ylabel='TPR')
axes[0].legend(loc='lower right', fontsize=8.5)

axes[1].axhline(y_test_ml.mean(), color='k', ls='--', lw=1, label=f"Azar ({y_test_ml.mean():.3f})")
axes[1].set(title='Curvas Precision-Recall (Test)', xlabel='Recall', ylabel='Precision')
axes[1].legend(loc='lower left', fontsize=8.5)

axes[2].plot([0, 1], [0, 1], 'k--', lw=1, label='Calibración perfecta')
axes[2].set(title='Reliability Diagrams (Test, 10 cuantiles)', xlabel='Prob. predicha', ylabel='Frec. observada')
axes[2].legend(loc='upper left', fontsize=8.5)
plt.tight_layout()
plt.show()
"""))

    # 3.6 Importancia MDI vs Permutation y Ablación DEBTINC_is_missing
    cells.append(nbf.v4.new_markdown_cell(r"""### 3.6 Diagnóstico de Importancia Predictiva (MDI vs. Permutation) y Ablación de `DEBTINC_is_missing`
Este análisis prepara el puente crítico hacia la **Fase 4 (Inferencia Causal)**: medimos qué variables dominan la predicción observacional frente a la variable de intervención `TREATMENT`, y auditamos cómo explotan los modelos el patrón de omisión MAR/MNAR detectado en Fase 1.
"""))

    cells.append(nbf.v4.new_code_cell(r"""imp_df = compute_feature_importance_comparison(rf_res['best_model'], X_train_ml, X_test_ml, y_test_ml)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ord_mdi = imp_df.sort_values('MDI').index
axes[0].barh(ord_mdi, imp_df.loc[ord_mdi, 'MDI'], color='#339af0')
axes[0].set_title('MDI (Impureza de Gini en Train)', fontweight='bold')

ord_perm = imp_df.sort_values('Permutacion').index
axes[1].barh(ord_perm, imp_df.loc[ord_perm, 'Permutacion'], xerr=imp_df.loc[ord_perm, 'Permutacion_SD'], color='#20c997')
axes[1].set_title('Permutation Importance (Caída AUC en Test)', fontweight='bold')
plt.tight_layout()
plt.show()

display(imp_df[['MDI', 'Rank_MDI', 'Permutacion', 'Rank_Perm', 'Valores_distintos']].round(4))

ablation_df = run_debtinc_missing_ablation(X_train_ml, y_train_ml, X_test_ml, y_test_ml, rf_res['best_params'])
print("\n--- ABLACIÓN DE DEBTINC_is_missing Y DEBTINC ---")
display(ablation_df.round(4))
"""))

    return cells
