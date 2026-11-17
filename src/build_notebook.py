"""
src/build_notebook.py
Genera y ejecuta el Notebook Maestro Reproducible TP_Integrador_Final.ipynb
integrando la Gobernanza de Datos (Contrato), Fase 1 y Fase 2.
"""

import os

import nbformat as nbf
from nbclient import NotebookClient


def build_and_execute():
    nb = nbf.v4.new_notebook()
    cells = []

    # ==========================================================================
    # ENCABEZADO & GOBERNANZA BASE
    # ==========================================================================
    cells.append(nbf.v4.new_markdown_cell("""# 🏆 Decision Intelligence Capstone: Modelado Predictivo, Inferencia Causal y Sistemas de Decisión en Producción
**Cátedra:** Estadística Aplicada y Modelos de Regresión (EAyMR 2026)  
**Institución:** Pontificia Universidad Católica Argentina (UCA) — Facultad de Ingeniería y Ciencias Agrarias  
**Profesor Titular:** Prof. Ivan Gonzalo Vilaboa  
**Integrantes y Roles Scrum:**
- 👑 **Project Lead, Scrum Master & Econometría (Fases 1–2):** German Sittner
- ⚙ **Backend & Machine Learning Engineer (Fase 3):** Martín Pena
- 🧠 **Causal Inference & Causal ML Engineer (Fase 4):** Santiago Salvatore
- 🔍 **QA, Validation & UI/UX Deployment (Transversal & Fase 5):** Equipo EAyMR (Validación cruzada `pytest`)

---

## 📌 Base del Proyecto: Contrato de Datos y Partición Oficial
Todos los análisis de este proyecto se sustentan en el contrato unificado definido en `src/config.py` y generado por `src/data_pipeline.py` sobre el dataset real **HMEQ** ($N = 5.960$, 13 variables originales):
- **Target Primario ($Y$):** `BAD` ($1$ = Default/Moroso, $0$ = Cumplidor).
- **Target Continuo ($Y_{\\text{cont}}$):** `LOAN` (monto de crédito solicitado).
- **Tratamiento Causal ($T$):** `TREATMENT` ($1$ = Consolidación de Deuda `DebtCon`, $0$ = Mejoras del Hogar `HomeImp`).
- **Factor ANOVA ($k \\ge 3$):** `JOB` (6 categorías ocupacionales).
- **Mediador Post-Tratamiento ($M$):** `LOAN` (excluido del ajuste causal Backdoor/DML).
- **Partición Oficial:** `data/processed/train.csv` ($80\\%$, $N = 4.768$) y `data/processed/test.csv` ($20\\%$, $N = 1.192$), imputada sin fuga de datos (*No Data Leakage*).
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

from src.config import (
    DATA_RAW_PATH, TRAIN_PATH, TEST_PATH, TARGET_BINARY, TARGET_CONTINUOUS,
    TREATMENT_VAR, ANOVA_FACTOR, ANOVA_VARIABLE, NUMERICAL_FEATURES
)
from src.phase1_inference import (
    load_phase1_datasets, analyze_missing_mechanism, compute_multidimensional_eda,
    run_hypothesis_contrasts, run_anova_and_diagnostics
)
from src.phase2_econometrics import (
    load_phase2_splits, fit_ols_reference, audit_gauss_markov,
    detect_influential_points, compare_robust_regressors, calculate_vif,
    run_pca_no_leakage
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

df_raw, train_df = load_phase1_datasets()
test_df = pd.read_csv(TEST_PATH)
print(f"Dataset crudo: {df_raw.shape} | Train oficial: {train_df.shape} | Test oficial: {test_df.shape}")
"""))

    # ==========================================================================
    # FASE 1: INFERENCIA ESTADÍSTICA CLÁSICA, EDA, CONTRASTES Y ANOVA
    # ==========================================================================
    cells.append(nbf.v4.new_markdown_cell("""---

## 🔬 FASE 1: Inferencia Estadística Clásica, Diagnósticos y ANOVA
*(Articulación directa con Unidades 1 y 2)*

### 1.1 Diagnóstico Matemático del Mecanismo de Pérdida de Datos (MCAR vs. MAR/MNAR)
Verificamos el patrón de omisión sobre el dataset original (`hmeq.csv`) mediante un modelo logit sobre la ausencia de `DEBTINC` ($21.26\\%$ de nulos):
$$\\text{logit}(P(\\text{DEBTINC is missing})) = \\beta_0 + \\beta_1 \\text{LOAN} + \\beta_2 \\text{VALUE} + \\beta_3 \\text{BAD}$$
"""))

    cells.append(nbf.v4.new_code_cell("""missing_res = analyze_missing_mechanism(df_raw)
display(missing_res['missing_table'][missing_res['missing_table']['Nulos'] > 0])
print(missing_res['logit_debtinc_missing'].summary().tables[1])
print(f"¿Es MCAR?: {missing_res['is_mcar']} | Odds Ratio de omisión para BAD=1: {missing_res['odds_ratio_bad']:.2f}x")
"""))

    cells.append(nbf.v4.new_markdown_cell("""**Conclusión:** El coeficiente de `BAD` es $z = 34.74$ ($p < 10^{-16}$, $OR \\approx 16.3$). Se rechaza formalmente el mecanismo MCAR a favor de **MAR/MNAR**. Por ello, en el pipeline oficial preservamos `DEBTINC_is_missing` e imputamos medianas aprendidas únicamente sobre Train.

---

### 1.2 EDA Multidimensional: Distribuciones KDE, Boxplots y Correlación Pearson vs. Spearman
"""))

    cells.append(nbf.v4.new_code_cell("""eda_res = compute_multidimensional_eda(train_df)
display(eda_res['descriptivos'])

# Visualización Multidimensional: KDE y Boxplots por BAD
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
vars_plot = ['LOAN', 'VALUE', 'DEBTINC']
for i, col in enumerate(vars_plot):
    sns.kdeplot(data=train_df, x=col, hue=TARGET_BINARY, fill=True, common_norm=False,
                palette={0: '#2ca02c', 1: '#d62728'}, ax=axes[0, i])
    axes[0, i].set_title(f'Densidad KDE de {col} según {TARGET_BINARY}', fontweight='bold')
    
    sns.boxplot(data=train_df, x=TARGET_BINARY, y=col, hue=TARGET_BINARY,
                palette={0: '#2ca02c', 1: '#d62728'}, legend=False, ax=axes[1, i])
    axes[1, i].set_title(f'Boxplot de {col} por {TARGET_BINARY}', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Matrices de Correlación Paramétrica (Pearson) vs No Paramétrica (Spearman)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
sns.heatmap(eda_res['corr_pearson'], annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, ax=axes[0], cbar=False)
axes[0].set_title('Correlación Lineal de Pearson (Train)', fontsize=12, fontweight='bold')

sns.heatmap(eda_res['corr_spearman'], annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title('Correlación de Rango de Spearman (Train)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.3 Contrastes Formales de Hipótesis ($H_0$ frente a $H_1$, $\\alpha = 0.05$)
1. **Contraste 1:** ¿El monto solicitado (`LOAN`) difiere entre cumplidores (`BAD=0`) y morosos (`BAD=1`)? ($H_0: \\mu_0 = \\mu_1$ vs $H_1: \\mu_0 \\neq \\mu_1$).
2. **Contraste 2:** ¿El ratio deuda/ingreso (`DEBTINC`) es mayor en consolidación de pasivos (`DebtCon`) frente a reformas (`HomeImp`)? ($H_0: \\mu_{\\text{DebtCon}} = \\mu_{\\text{HomeImp}}$ vs $H_1: \\mu_{\\text{DebtCon}} > \\mu_{\\text{HomeImp}}$).
"""))

    cells.append(nbf.v4.new_code_cell("""contrasts = run_hypothesis_contrasts(df_raw, alpha=0.05)

for key, res in contrasts.items():
    print(f"=== {res['titulo']} ===")
    print(f"Diferencia de Medias : {res['diff_mean']:,.4f} | IC 95%: [{res['diff_ci95'][0]:,.4f}, {res['diff_ci95'][1]:,.4f}]")
    print(f"Welch t-test         : t = {res['t_stat']:.4f}, p-valor = {res['p_value_welch']:.4e}")
    print(f"Mann-Whitney U       : U = {res['u_stat']:.1f}, p-valor = {res['p_value_mwu']:.4e}")
    print(f"Magnitud (Cohen's d) : {res['cohen_d']:.4f} | Rechaza H0 (alpha=0.05): {res['reject_h0']}\\n")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.4 ANOVA de un Factor (`LOAN` ~ `JOB`), Desafío Shapiro-Wilk en $N$ Grande y Remediación con Welch ANOVA
Evaluamos si el monto de crédito varía entre las 6 ocupaciones (`JOB`). Con $N \\ge 1.500$, Shapiro-Wilk rechaza $H_0$ ante perturbaciones microscópicas; por ello auditamos asimetría ($[-1, 1]$), curtosis, Q-Q plots y el **test de Levene** para decidir entre ANOVA clásico y **Welch's ANOVA**.
"""))

    cells.append(nbf.v4.new_code_cell("""anova_res = run_anova_and_diagnostics(train_df)
display(anova_res['diagnostico_grupos'])

print(f"Test de Levene (homocedasticidad): W = {anova_res['levene_test']['W_stat']:.4f}, p = {anova_res['levene_test']['p_value']:.4e}")
print(f"Welch's ANOVA (robusto)          : F* = {anova_res['welch_anova']['F_welch']:.4f}, "
      f"df1 = {anova_res['welch_anova']['df1']}, df2 = {anova_res['welch_anova']['df2']:.2f}, p = {anova_res['welch_anova']['p_value']:.4e}")
print(f"Kruskal-Wallis (no paramétrico)  : H = {anova_res['kruskal_wallis']['H_stat']:.4f}, p = {anova_res['kruskal_wallis']['p_value']:.4e}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Q-Q Plots por grupo ocupacional y Boxplot comparativo + Post-hoc Tukey HSD
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
sns.boxplot(data=train_df, x=ANOVA_FACTOR, y=ANOVA_VARIABLE, hue=ANOVA_FACTOR, palette='viridis', legend=False, ax=axes[0])
axes[0].set_title('Distribución de LOAN por Categoría Ocupacional (JOB)', fontweight='bold')

anova_res['tukey_posthoc'].plot_simultaneous(ax=axes[1])
axes[1].set_title('Comparaciones Múltiples Post-Hoc: Tukey HSD (IC 95%)', fontweight='bold')
plt.tight_layout()
plt.show()

print("Comparaciones Post-Hoc Welch con corrección de Bonferroni:")
display(anova_res['bonferroni_welch_posthoc'])
"""))

    # ==========================================================================
    # FASE 2: DIAGNÓSTICO DE GAUSS-MARKOV, COOK'S D Y PCA SIN LEAKAGE
    # ==========================================================================
    cells.append(nbf.v4.new_markdown_cell("""---

## 📐 FASE 2: Diagnóstico Econométrico de Gauss-Markov, Puntos Influyentes y Reducción PCA
*(Articulación directa con Unidad 3)*

### 2.1 Ajuste del Modelo Lineal Clásico de Referencia (MCO) sobre `train.csv`
Estimamos el modelo de regresión lineal múltiple sobre las 9 covariables numéricas basales para explicar el monto de préstamo solicitado (`LOAN`):
$$Y = \\beta_0 + \\beta_1 X_1 + \\dots + \\beta_9 X_9 + \\varepsilon$$
"""))

    cells.append(nbf.v4.new_code_cell("""tr_df, te_df, pred_cols = load_phase2_splits()
X_train_ols, y_train_ols = tr_df[pred_cols], tr_df[TARGET_CONTINUOUS]
X_test_ols, y_test_ols = te_df[pred_cols], te_df[TARGET_CONTINUOUS]

ols_model = fit_ols_reference(X_train_ols, y_train_ols)
print(ols_model.summary())
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.2 Auditoría Exhaustiva de Supuestos de Gauss-Markov
Contrastamos homocedasticidad (**Breusch-Pagan** y **White**), normalidad de residuos (**Jarque-Bera** y Q-Q Plot) e independencia (**Durbin-Watson**).
"""))

    cells.append(nbf.v4.new_code_cell("""gm_audit = audit_gauss_markov(ols_model)
print(f"Breusch-Pagan : LM = {gm_audit['breusch_pagan']['lm_stat']:.4f}, p = {gm_audit['breusch_pagan']['p_value']:.4e} | Homocedástico: {gm_audit['breusch_pagan']['is_homoscedastic']}")
print(f"Test de White : LM = {gm_audit['white_test']['lm_stat']:.4f}, p = {gm_audit['white_test']['p_value']:.4e} | Homocedástico: {gm_audit['white_test']['is_homoscedastic']}")
print(f"Jarque-Bera   : JB = {gm_audit['jarque_bera']['jb_stat']:.2f}, p = {gm_audit['jarque_bera']['p_value']:.4e}, Skew = {gm_audit['jarque_bera']['skewness']:.2f}, Kurt = {gm_audit['jarque_bera']['kurtosis']:.2f}")
print(f"Durbin-Watson : DW = {gm_audit['durbin_watson']['dw_stat']:.4f} | Independencia serial OK: {gm_audit['durbin_watson']['is_independent']}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].scatter(ols_model.fittedvalues, ols_model.resid, alpha=0.25, s=15, color='#1f77b4')
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_title('Residuos vs. Valores Ajustados (Heterocedasticidad)', fontweight='bold')
axes[0].set_xlabel('Valores Ajustados ($\\hat{y}$)')
axes[0].set_ylabel('Residuos Ordinarios ($e_i$)')

sm.qqplot(ols_model.resid, line='45', fit=True, ax=axes[1])
axes[1].set_title('Q-Q Plot de Residuos frente a Distribución Normal', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.3 Detección de Observaciones Extremas, Distancia de Cook ($D_i$) y Regresión Robusta
Descomponemos outliers de respuesta ($|r_i| > 3$), apalancamiento ($h_{ii} > \\frac{2(p+1)}{n}$) e influencia global ($D_i = \\frac{e_i^2}{(p+1)\\text{MSE}} \\left[\\frac{h_{ii}}{(1-h_{ii})^2}\\right]$), y evaluamos la remediación mediante **Huber** y **RANSAC** sobre el conjunto de prueba (`test.csv`).
"""))

    cells.append(nbf.v4.new_code_cell("""infl_res = detect_influential_points(ols_model, X_train_ols)
print(f"Outliers de respuesta (|r_i| > 3)     : {infl_res['count_outliers_resid']}")
print(f"Alto apalancamiento (h_ii > {infl_res['thresh_leverage']:.5f}) : {infl_res['count_high_leverage']}")
print(f"Puntos influyentes (Cook's D > 4/n)  : {infl_res['count_high_cooks_4n']} (Max D_i = {infl_res['max_cooks_d']:.4f})")

comp_robust = compare_robust_regressors(X_train_ols, y_train_ols, X_test_ols, y_test_ols, ols_model)
display(comp_robust)

fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(infl_res['leverage'], infl_res['cooks_distance'], c=np.abs(infl_res['stud_resids']), cmap='plasma', alpha=0.55, s=20)
ax.axvline(infl_res['thresh_leverage'], color='orange', linestyle='--', label=f"Leverage crítico ({infl_res['thresh_leverage']:.4f})")
ax.axhline(infl_res['thresh_cook_4n'], color='red', linestyle='--', label=f"Cook's D crítico 4/n ({infl_res['thresh_cook_4n']:.5f})")
plt.colorbar(sc, ax=ax, label='|Residuo Estudentizado|')
ax.set_title("Distancia de Cook ($D_i$) frente a Apalancamiento ($h_{ii}$)", fontweight='bold')
ax.set_xlabel('Leverage ($h_{ii}$)')
ax.set_ylabel("Distancia de Cook ($D_i$)")
ax.legend()
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.4 Diagnóstico de Multicolinealidad (VIF)
Calculamos $\\text{VIF}_j = \\frac{1}{1 - R_j^2}$ sobre las covariables de entrenamiento para auditar redundancia estructural.
"""))

    cells.append(nbf.v4.new_code_cell("""df_vif = calculate_vif(X_train_ols)
display(df_vif)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.5 Reducción de Dimensionalidad con PCA (Estrictamente Ajustado en Train)
Siguiendo la regla anti-leakage, `StandardScaler` y `PCA` se ajustan **únicamente sobre `train.csv`** y transforman `test.csv` de manera ciega. Retenemos componentes bajo el **criterio de Kaiser ($\\lambda_k > 1$)**.
"""))

    cells.append(nbf.v4.new_code_cell("""pca_res = run_pca_no_leakage(tr_df, te_df, pred_cols)

pca_table = pd.DataFrame({
    'Autovalor (Lambda)': pca_res['eigenvalues'],
    'Varianza (%)': pca_res['var_exp_ratio'] * 100,
    'Varianza Acumulada (%)': pca_res['cum_var_exp'] * 100,
    'Retenido (Kaiser > 1)': pca_res['eigenvalues'] > 1.0
}, index=[f"PC{i+1}" for i in range(len(pca_res['eigenvalues']))])
display(pca_table)
print(f"Componentes retenidos según Kaiser (Lambda > 1): {pca_res['kaiser_retained_count']}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Scree Plot, Varianza Acumulada y Matriz de Cargas (Loadings)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(range(1, len(pca_res['eigenvalues']) + 1), pca_res['eigenvalues'], 'o-', color='#1f77b4', lw=2)
axes[0].axhline(1.0, color='red', linestyle='--', label='Kaiser (λ = 1.0)')
axes[0].set_title('Scree Plot (Train Set)', fontweight='bold')
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Autovalor (λ)')
axes[0].legend()

axes[1].bar(range(1, len(pca_res['var_exp_ratio']) + 1), pca_res['var_exp_ratio'] * 100, alpha=0.5, color='#2ca02c', label='Individual')
axes[1].step(range(1, len(pca_res['cum_var_exp']) + 1), pca_res['cum_var_exp'] * 100, where='mid', color='black', lw=2, label='Acumulada')
axes[1].set_title('Varianza Explicada (%)', fontweight='bold')
axes[1].set_xlabel('Componentes')
axes[1].legend()

sns.heatmap(pca_res['loadings'].iloc[:, :3], annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[2])
axes[2].set_title('Cargas Factoriales (PC1-PC3)', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Proyección 2D en Test (ciega) diferenciada por Default (BAD)
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(
    pca_res['X_test_pca'][:, 0], pca_res['X_test_pca'][:, 1],
    c=te_df[TARGET_BINARY], cmap='coolwarm', alpha=0.65, s=25
)
ax.set_title('Proyección PCA en Test Set (PC1 vs PC2) coloreada por BAD', fontweight='bold')
ax.set_xlabel(f"PC1 - Capacidad Patrimonial ({pca_res['var_exp_ratio'][0]*100:.1f}%)")
ax.set_ylabel(f"PC2 - Estrés Financiero y Buró ({pca_res['var_exp_ratio'][1]*100:.1f}%)")
plt.colorbar(scatter, ax=ax, label='BAD (0=Cumplidor, 1=Default)')
plt.tight_layout()
plt.show()
"""))

    # ==========================================================================
    # FASE 3 Y FASE 4 + ENSAYO CRÍTICO UNIFICADO
    # ==========================================================================
    from src.build_phase3_notebook import get_phase3_cells
    from src.build_phase4_notebook import get_phase4_cells

    cells.extend(get_phase3_cells())
    cells.extend(get_phase4_cells())

    nb['cells'] = cells

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'TP_Integrador_Final.ipynb'))
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"Notebook unificado generado en {out_path} ({len(cells)} celdas). Ejecutando de punta a punta...")
    client = NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': os.path.dirname(out_path)}})
    client.execute()

    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print("Notebook TP_Integrador_Final.ipynb (Fases 1 a 4 + Ensayo Crítico) ejecutado de punta a punta sin errores!")


if __name__ == "__main__":
    build_and_execute()
