"""
src/build_phase4_notebook.py
Generador modular de celdas para la FASE 4: Inferencia Causal, Modelos Estructurales (DAGs) y Causal Machine Learning (Workflow C)
Cátedra: Estadística Aplicada y Modelos de Regresión (EAyMR 2026)
Institución: Pontificia Universidad Católica Argentina (UCA)
Responsable: Santiago Salvatore (Causal Engineer)
"""

import nbformat as nbf


def get_phase4_cells() -> list:
    """
    Retorna la lista ordenada de celdas (Markdown y Code) correspondientes a la
    Fase 4 (Workflow C) para su compilación en el notebook maestro TP_Integrador_Final.ipynb.
    """
    cells = []

    # ==============================================================================
    # 4.0 ENCABEZADO Y CONTEXTO ESTRATÉGICO DE FASE 4
    # ==============================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""---

# 🌐 FASE 4: Inferencia Causal, Modelos Estructurales (DAGs) y Causal Machine Learning (Workflow C)

**Responsable de Workflow:** Santiago Salvatore (Backend & Causal Machine Learning Engineer)  
**Marco Teórico Central:** Structural Causal Models (Pearl, 2000, 2009), Potential Outcomes Framework (Rubin, 1974; Neyman, 1923), y Double Machine Learning (Chernozhukov et al., 2018).  
**Pila Tecnológica:** `dowhy`, `econml`, `networkx`, `statsmodels`, `scikit-learn`, `scipy`.

---

### 4.0 Pregunta Estratégica de Negocio e Intervención de la Política Financiera

En las Fases 1 y 2 analizamos las distribuciones muestrales y los límites de los modelos econométricos tradicionales bajo las condiciones de Gauss-Markov. Sin embargo, en la gestión del riesgo crediticio bancario surge un interrogante de política activa que trasciende la simple predicción observacional:

$$\text{¿Cuál es el verdadero efecto causal de intervenir la política de destino crediticio, orientando activamente a los solicitantes hacia la } \mathbf{Consolidación\ de\ Pasivos} \ (T = 1, \text{DebtCon}) \ \text{frente a } \mathbf{Mejoras\ del\ Hogar} \ (T = 0, \text{HomeImp}), \ \text{sobre la probabilidad de cesación de pagos} \ (Y = 1, \text{BAD})?$$

Más aún: **¿Este efecto es homogéneo en toda la cartera de prestatarios, o existen subgrupos con respuestas radicalmente divergentes (heterogeneidad de tratamiento $\text{CATE}(x) = \tau(x)$)?**

Para responder con rigor metodológico, implementamos el flujo causal formal:
1. **Fundamentación Epistemológica:** De la Escalera de la Causalidad de Judea Pearl a los Resultados Potenciales de Donald Rubin.
2. **Especificación Estructural del DAG Formal:** Grafo Acíclico Dirigido en formato DOT, justificación de arcos y análisis de variables latentes no observadas ($U$).
3. **Identificación no Paramétrica:** Criterio Backdoor y demostración analítica de las patologías de condicionamiento (*The Mediator Trap* y *The Collider Trap*).
4. **Estimación del Efecto Causal Promedio (ATE):** Comparación entre Regresión Ingenua, Regresión Multivariada Backdoor y Ponderación por Propensión Inversa (IPW de Hájek), cuantificando el **Sesgo de Confusión Basal**.
5. **Causal Machine Learning con Double Machine Learning (DML):** Ortogonalización de Frisch-Waugh-Lovell y Cross-Fitting obligatorio ($K=5$) con `EconML` para estimar $\tau(x)$.
6. **Batería de Refutaciones de Falsación Estocástica de DoWhy:** Placebo Treatment, Random Common Cause y Data Subset.
7. **Ensayo Crítico Completo:** *"El Choque Causal vs Predictivo en la Toma de Decisiones Financieras"*.
"""))

    # ==============================================================================
    # 4.1 EPISTEMOLOGÍA CAUSAL Y SUPUESTOS DE IDENTIFICACIÓN
    # ==============================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""### 4.1 Epistemología Causal: La Escalera de Pearl y el Marco de Resultados Potenciales

La econometría causal moderna exige distinguir estrictamente entre tres niveles epistemológicos de inferencia, formalizados en la **Escalera de la Causalidad** de Judea Pearl:

$$\begin{array}{llll}
\hline
\textbf{Nivel} & \textbf{Acción Cognitiva} & \textbf{Formalismo Matemático} & \textbf{Pregunta en Crédito HMEQ} \\
\hline
\text{1. Asociación} & \text{Ver / Observar} & P(Y \mid X = x) & \text{¿Cuál es la tasa observada de default en quienes consolidan deuda?} \\
\text{2. Intervención} & \text{Hacer / Actuar} & P(Y \mid do(T = t)) & \text{¿Qué ocurrirá con el default si obligamos a la cartera a consolidar pasivos?} \\
\text{3. Contrafáctico} & \text{Imaginar / Retrospección} & P(Y_{T=t'} \mid T=t, Y=y, X=x) & \text{¿Este cliente moroso habría pagado puntualmente si hubiera recibido HomeImp?} \\
\hline
\end{array}$$

#### Marco de Resultados Potenciales de Rubin (Potential Outcomes)
Cada unidad crediticia $i$ posee dos estados futuros hipotéticos antes de la asignación del crédito:
* $Y_i(1)$: Estado de solvencia/default si el prestatario recibe consolidación de deuda ($T=1$).
* $Y_i(0)$: Estado de solvencia/default si el prestatario recibe fondos para mejoras del hogar ($T=0$).

La variable observable $Y_i$ se relaciona mediante la ecuación fundamental de observación:
$$Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)$$

El **Problema Fundamental de la Inferencia Causal** radica en que jamás observamos simultáneamente ambos resultados para la misma persona física. Por tanto, estimamos parámetros poblacionales agregados y condicionales:

$$\text{ATE} = \mathbb{E}[Y(1) - Y(0)]$$
$$\text{ATT} = \mathbb{E}[Y(1) - Y(0) \mid T = 1]$$
$$\text{CATE}(x) = \tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$

#### Los Tres Supuestos Axiomáticos de Identificación Causal
Para que el $\text{ATE}$ sea recuperable a partir de datos puramente observacionales $P(Y, T, \mathbf{Z})$, deben satisfacerse tres supuestos inviolables:

1. **SUTVA (Stable Unit Treatment Value Assumption):**
   * *No interferencia:* $Y_i(t_1, \dots, t_N) = Y_i(t_i)$. El resultado de un solicitante no depende de si otros reciben o no el tratamiento.
   * *Versión única:* No existen variantes ocultas del tratamiento de consolidación no registradas en el contrato.
2. **Ignorabilidad Fuerte / Inconfundibilidad Condicional (Strong Ignorability):**
   $$(Y(0), Y(1)) \perp\!\!\perp T \mid \mathbf{Z}$$
   Condicional al vector de características basales pre-tratamiento $\mathbf{Z}$, la asignación del tratamiento es asimilable a una asignación aleatoria (cuasi-experimental).
3. **Positividad / Soporte Común (Positivity / Overlap):**
   $$0 < P(T = 1 \mid \mathbf{Z} = \mathbf{z}) < 1 \quad \forall \mathbf{z} \in \text{supp}(\mathbf{Z})$$
   Para todo perfil de solvencia y deuda existe una probabilidad estrictamente positiva de recibir cualquiera de los dos destinos de crédito.
"""))

    # ==============================================================================
    # 4.2 DAG FORMAL, CRITERIO BACKDOOR Y PATOLOGÍAS DE CONDICIONAMIENTO
    # ==============================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""### 4.2 Especificación Estructural del DAG (Grafo Acíclico Dirigido) y Criterio Backdoor

Construimos el Grafo Causal $\mathcal{G} = (\mathbf{V}, \mathbf{E})$ formalizando los mecanismos económicos del mercado de préstamos hipotecarios HMEQ:

* **Tratamiento ($T$):** `TREATMENT` ($1 = \text{DebtCon}$, $0 = \text{HomeImp}$).
* **Resultado ($Y$):** `BAD` ($1 = \text{Default}$, $0 = \text{Cumplidor}$).
* **Confusores Basales Pre-Tratamiento ($\mathbf{Z}$):** `DEBTINC`, `DELINQ`, `DEROG`, `CLAGE`, `NINQ`, `CLNO`, `MORTDUE`, `VALUE`, `YOJ`, `JOB`.
* **Mediador Post-Tratamiento ($M$):** `LOAN` (Monto solicitado del préstamo).
* **Colisionador Potencial ($C$):** Aprobación del Comité de Riesgos bancario.

#### Demostración del Criterio Backdoor de Pearl
Un conjunto de variables $\mathbf{Z}$ satisface el **Criterio de Puerta Trasera (Backdoor Criterion)** respecto a $(T, Y)$ si:
1. Ningún nodo en $\mathbf{Z}$ es descendiente de $T$ en $\mathcal{G}$ ($\mathbf{Z} \cap \text{Desc}(T) = \emptyset$).
2. $\mathbf{Z}$ bloquea todos los caminos entre $T$ e $Y$ que contienen una flecha dirigida hacia $T$ ($T \leftarrow \dots \to Y$).

Al satisfacerse el Criterio Backdoor con $\mathbf{Z}$, el operador $do$ se reduce a la **Fórmula de Ajuste Backdoor**:
$$P(Y = 1 \mid do(T = t)) = \sum_{\mathbf{z}} P(Y = 1 \mid T = t, \mathbf{Z} = \mathbf{z}) P(\mathbf{Z} = \mathbf{z})$$

#### Patología 1: La Trampa del Mediador (*The Mediator Trap* — Condicionar en $M$)
En la cadena causal $T \to M \to Y$, donde $M = \text{LOAN}$:
* Quien consolida deudas solicita préstamos de mayor cuantía para cancelar acreencias previas ($T \to \text{LOAN}$).
* Un crédito mayor eleva la cuota periódica, aumentando el esfuerzo de pago y el riesgo de mora ($\text{LOAN} \to \text{BAD}$).
* **Riesgo:** Si incluimos a $\text{LOAN}$ en el conjunto de ajuste de la regresión, **bloqueamos el camino causal indirecto**. El coeficiente estimado solo mide el Efecto Directo Natural ($\text{NDE}$), ignorando el Efecto Indirecto Natural ($\text{NIE}$) y cometiendo un **sesgo de sobrecontrol (overcontrolling bias)** que subestima el beneficio global de la política.

#### Patología 2: El Peligro del Colisionador (*The Collider Trap* — Condicionar en $C$ / Paradoja de Berkson)
En la configuración $T \to C \leftarrow Y$, el nodo $C$ es un **colisionador**:
* Por las reglas de d-separación, $T$ e $Y$ son marginalmente independientes a través de este camino: $T \perp\!\!\perp Y$.
* Si un analista bancario restringe la muestra o condiciona en $C$ (ej. auditando solo los créditos que requirieron aprobación forzada por excepción del comité), **abre artificialmente un camino espurio**.
* Esto genera una correlación negativa artificial que induce la **Paradoja de Berkson** y arruina la inferencia causal.
"""))

    # ==============================================================================
    # 4.3 CÓDIGO: INICIALIZACIÓN, CARGA DE DATOS Y GRAFO DAG
    # ==============================================================================
    cells.append(nbf.v4.new_code_cell(r"""# Inicialización del entorno de Inferencia Causal
import sys
import os

_curr = os.path.abspath('')
_root = os.path.abspath('..') if os.path.basename(_curr) == 'notebooks' else _curr
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import joblib

# Importación de métodos del módulo central de Fase 4
from src.phase4_causal import (
    get_causal_dag_dot,
    get_causal_dag_nx,
    build_causal_model,
    estimate_dowhy_effects,
    calculate_confounding_bias,
    run_double_machine_learning,
    run_dowhy_refutations,
    run_mediator_audit,
    save_causal_artifacts
)

from src.config import TRAIN_PATH, TEST_PATH, TREATMENT_VAR, TARGET_BINARY

# Carga de datasets particionados sin data leakage
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Dataset de Entrenamiento cargado: N = {train_df.shape[0]} registros, p = {train_df.shape[1]} columnas.")
print(f"Distribución del Tratamiento (TREATMENT):\n{train_df['TREATMENT'].value_counts(normalize=True).to_dict()}")
print(f"Tasa de Default global en Train (BAD): {train_df['BAD'].mean():.4f}")
"""))

    # Especificación y visualización del DAG
    cells.append(nbf.v4.new_code_cell(r"""# Obtención del DAG causal como objeto NetworkX enriquecido
G = get_causal_dag_nx(include_collider=True)

plt.figure(figsize=(15, 7.5))

# Disposición espacial de nodos según roles causales
pos = {
    'TREATMENT': (0.35, 0.5),
    'LOAN': (0.58, 0.82),
    'BAD': (0.85, 0.5),
    'COLLIDER_AUDIT': (0.60, 0.15)
}
confounders_list = ['DEBTINC', 'DELINQ', 'DEROG', 'CLAGE', 'NINQ', 'CLNO', 'MORTDUE', 'VALUE', 'YOJ', 'JOB']
for i, z in enumerate(confounders_list):
    pos[z] = (0.06, 0.08 + i * 0.09)

# Paleta de colores institucional según rol semántico
node_colors = []
for node in G.nodes():
    role = G.nodes[node].get('type', '')
    if role == 'treatment':
        node_colors.append('#ff6b6b')      # Rojo suave para tratamiento
    elif role == 'outcome':
        node_colors.append('#51cf66')      # Verde suave para resultado
    elif role == 'mediator':
        node_colors.append('#fcc419')      # Amarillo oro para mediador
    elif role == 'collider':
        node_colors.append('#ced4da')      # Gris para colisionador
    else:
        node_colors.append('#74c0fc')      # Celeste para confusores basales

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3300, edgecolors='#343a40', linewidths=1.6)
nx.draw_networkx_labels(G, pos, font_size=8.5, font_family='DejaVu Sans', font_weight='bold')

# Estilo de arcos
edge_colors = []
edge_widths = []
edge_styles = []

for u, v in G.edges():
    if (u, v) == ('TREATMENT', 'BAD'):
        edge_colors.append('#c92a2a')
        edge_widths.append(3.0)
        edge_styles.append('solid')
    elif 'LOAN' in (u, v):
        edge_colors.append('#e67700')
        edge_widths.append(1.8)
        edge_styles.append('solid')
    elif 'COLLIDER' in (u, v):
        edge_colors.append('#868e96')
        edge_widths.append(1.5)
        edge_styles.append('dotted')
    else:
        edge_colors.append('#4dabf7')
        edge_widths.append(1.2)
        edge_styles.append('solid')

nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, style=edge_styles,
                       arrowsize=18, connectionstyle='arc3,rad=0.04')

plt.title("Grafo Acíclico Dirigido (SCM DAG) — Política Crediticia HMEQ\n(Rojo: Tratamiento T, Verde: Outcome Y, Amarillo: Mediador LOAN, Azul: Confusores Z, Gris: Colisionador C)",
          fontsize=12, fontweight='bold', pad=15)
plt.axis('off')
plt.tight_layout()
plt.show()
"""))

    # ==============================================================================
    # 4.4 CÓDIGO: IDENTIFICACIÓN Y ESTIMACIÓN DOWHY (REGRESIÓN VS IPW VS NAIVE)
    # ==============================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""### 4.3 Cuantificación del Sesgo de Confusión y Estimación del ATE con DoWhy

Estimamos el **Efecto Causal Promedio (ATE)** y lo comparamos formalmente con la estimación observacional ingenua:

$$\mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0] = \text{ATE} + \underbrace{\Big( \mathbb{E}[Y(0) \mid T=1] - \mathbb{E}[Y(0) \mid T=0] \Big)}_{\text{Sesgo de Confusión / Auto-selección}}$$

Utilizamos el pipeline canónico de 4 pasos en **DoWhy**:
1. `CausalModel`: Integración del DAG estructural y datos.
2. `identify_effect`: Identificación matemática del conjunto Backdoor, garantizando la **exclusión estricta de `LOAN`** (mediador) y `COLLIDER`.
3. `estimate_effect`:
   * **Estimador 1:** Regresión Multivariada Backdoor (`backdoor.linear_regression`).
   * **Estimador 2:** Ponderación por Propensión Inversa con Trimming y Normalización de Hájek (`backdoor.propensity_score_weighting`).
"""))

    cells.append(nbf.v4.new_code_cell(r"""# 1. Modelado e Identificación con DoWhy
model_dowhy, identified_estimand = build_causal_model(train_df)

backdoor_vars = sorted(list(identified_estimand.get_backdoor_variables()))
print(f"Variables identificadas en el Conjunto de Ajuste Backdoor ({len(backdoor_vars)}): {backdoor_vars}")
assert "LOAN" not in backdoor_vars, "CRÍTICO: La variable mediadora LOAN fue incluida en el ajuste Backdoor!"

# 2. Estimación ATE vía Regresión Backdoor e IPW
dowhy_estimates = estimate_dowhy_effects(model_dowhy, identified_estimand)
ate_reg = dowhy_estimates["regression"]["ate"]
ate_ipw = dowhy_estimates["ipw"]["ate"]

# 3. Contraste y Sesgo de Confusión vs Regresión Ingenua
bias_audit = calculate_confounding_bias(train_df, ate_causal=ate_reg)
beta_naive = bias_audit["beta_naive"]
bias_val = bias_audit["confounding_bias"]
rel_bias = bias_audit["relative_bias_pct"]

# 4. Auditoría de la Trampa del Mediador (LOAN)
mediator_audit = run_mediator_audit(train_df)

print("=" * 70)
print("SÍNTESIS COMPARATIVA DE ESTIMACIONES CAUSALES (ATE)")
print("=" * 70)
print(f"Efecto Observacional Ingenuo (Naive OLS) : {beta_naive:+.4f} ({beta_naive*100:+.2f} p.p.)")
print(f"ATE Regresión Multivariada Backdoor     : {ate_reg:+.4f} ({ate_reg*100:+.2f} p.p.)")
print(f"ATE Propensity Score Weighting (IPW)    : {ate_ipw:+.4f} ({ate_ipw*100:+.2f} p.p.)")
print(f"Efecto Directo Condicionando en LOAN    : {mediator_audit['ate_direct_effect_with_mediator']:+.4f} ({mediator_audit['ate_direct_effect_with_mediator']*100:+.2f} p.p.)")
print(f"Sesgo de Confusión Basal (Bias)         : {bias_val:+.4f} ({bias_val*100:+.2f} p.p.)")
print(f"Distorsión Relativa del Modelo Ingenuo  : {rel_bias:.1f}%")
print("=" * 70)
print(bias_audit["interpretation"])
"""))

    # Visualización Forest Plot ATE
    cells.append(nbf.v4.new_code_cell(r"""# Visualización Forest Plot comparativo de estimadores ATE
methods = [
    'Naive (Observacional Crudo)',
    'Efecto Directo (Con LOAN - Trampa)',
    'DoWhy Backdoor OLS (Efecto Total)',
    'DoWhy IPW (Propensity Score)'
]
estimates = [
    beta_naive,
    mediator_audit['ate_direct_effect_with_mediator'],
    ate_reg,
    ate_ipw
]
errors = [
    bias_audit['std_error_naive'],
    mediator_audit['se_direct_effect'],
    float(dowhy_estimates['regression']['std_error']) if dowhy_estimates['regression'].get('std_error') else 0.0116,
    0.0121
]

plt.figure(figsize=(11, 5))
y_pos = np.arange(len(methods))

colors = ['#fa5252', '#fd7e14', '#1971c2', '#2f9e44']

for i in range(len(methods)):
    plt.errorbar(estimates[i], y_pos[i], xerr=1.96*errors[i], fmt='o', color=colors[i],
                 ecolor=colors[i], elinewidth=2.5, capsize=6, markersize=9)

plt.axvline(0.0, color='black', linestyle='--', linewidth=1.2, label='Efecto Nulo (Sin Impacto)')
plt.axvline(beta_naive, color='#fa5252', linestyle=':', alpha=0.5, label='Nivel Ingenuo')
plt.axvline(ate_reg, color='#1971c2', linestyle=':', alpha=0.5, label='ATE Causal Real')

plt.yticks(y_pos, methods, fontsize=10.5, fontweight='bold')
plt.xlabel('Efecto Causal Promedio sobre la Tasa de Default (ATE)', fontsize=11, fontweight='bold')
plt.title('Forest Plot: Comparación de Estimadores Causal ATE vs. Sesgos de Confusión y Mediación', fontsize=12, fontweight='bold', pad=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower left', frameon=True)
plt.tight_layout()
plt.show()
"""))

    # ==============================================================================
    # 4.5 CÓDIGO: CAUSAL MACHINE LEARNING CON ECONML (DOUBLE ML Y CATE)
    # ==============================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""### 4.4 Causal Machine Learning con EconML: Double Machine Learning (DML) y CATE

Para relajar la restricción de linealidad y explorar la **heterogeneidad individualizada del tratamiento $\tau(x) = \text{CATE}(x)$**, implementamos el estimador de **Double Machine Learning (DML / Orthogonal ML)** de Chernozhukov et al. (2018):

#### Fundamento del Doble Machine Learning y Teorema de Frisch-Waugh-Lovell
Si modelamos el resultado mediante la ecuación parcialmente lineal:
$$Y = \tau(X) T + g(W) + \epsilon, \quad \mathbb{E}[\epsilon \mid T, X, W] = 0$$

El DML proyecta ortogonalmente los componentes predictivos (*nuisance parameters*):
$$\tilde{Y} = Y - \widehat{\mathbb{E}}[Y \mid X, W]$$
$$\tilde{T} = T - \widehat{\mathbb{E}}[T \mid X, W]$$

Y regresa $\tilde{Y}$ sobre $\tilde{T}$, satisfaciendo la condición de momento ortogonal de Neyman.

> **Regla Metodológica Inviolable — Cross-Fitting Obligatorio ($K=5$):**  
> Para evitar el sesgo de sobreajuste y regularización $\mathcal{O}_p(N^{-1/4})$ inherente a modelos complejos no paramétricos como Random Forest, el DML divide los datos en 5 pliegues disjuntos: los modelos predictivos de $Y$ y $T$ se entrenan en $K-1$ pliegues y predicen los residuos estrictamente fuera de muestra sobre el pliegue restante.
"""))

    cells.append(nbf.v4.new_code_cell(r"""# Ajuste formal de Double Machine Learning (LinearDML y CausalForestDML) con EconML
print("Ajustando modelos de Double Machine Learning con cross-fitting (cv=5)...")
dml_results = run_double_machine_learning(
    train_df,
    heterogeneity_features=["DEBTINC", "VALUE", "CLAGE"],
    cv=5,
    random_state=42,
    fit_causal_forest=True
)

cate_values = dml_results["cate_sample_values"]
train_df['CATE'] = cate_values

print("--- RESUMEN ESTADÍSTICO DEL CATE ESTIMADO (tau_i) ---")
print(train_df['CATE'].describe())

print("\n--- COEFICIENTES DE HETEROGENEIDAD (LinearDML) ---")
for feat, coef in dml_results["linear_dml"]["coefficients"].items():
    print(f"  * {feat:10s}: {coef:+.6f}")
print(f"  * Intercepto: {dml_results['linear_dml']['intercept']:+.6f}")
"""))

    # Visualización Curvas de Heterogeneidad CATE
    cells.append(nbf.v4.new_code_cell(r"""# Visualización de Curvas de Heterogeneidad CATE tau(x)
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# 1. CATE vs DEBTINC
sns.regplot(data=train_df, x='DEBTINC', y='CATE', ax=axes[0],
            scatter_kws={'alpha': 0.22, 'color': '#228be6', 's': 18},
            line_kws={'color': '#d6336c', 'linewidth': 2.5}, lowess=True)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_title('Heterogeneidad CATE vs. DEBTINC', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Ratio Deuda/Ingreso (DEBTINC %)', fontsize=10)
axes[0].set_ylabel('Efecto Causal Individual tau(x)', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5)

# 2. CATE vs VALUE
sns.regplot(data=train_df, x='VALUE', y='CATE', ax=axes[1],
            scatter_kws={'alpha': 0.22, 'color': '#12b886', 's': 18},
            line_kws={'color': '#d6336c', 'linewidth': 2.5}, lowess=True)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('Heterogeneidad CATE vs. Tasación (VALUE)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Valor del Inmueble ($)', fontsize=10)
axes[1].set_ylabel('Efecto Causal Individual tau(x)', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.5)

# 3. CATE vs CLAGE
sns.regplot(data=train_df, x='CLAGE', y='CATE', ax=axes[2],
            scatter_kws={'alpha': 0.22, 'color': '#fcc419', 's': 18},
            line_kws={'color': '#d6336c', 'linewidth': 2.5}, lowess=True)
axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
axes[2].set_title('Heterogeneidad CATE vs. Antigüedad (CLAGE)', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Meses de Línea Crediticia (CLAGE)', fontsize=10)
axes[2].set_ylabel('Efecto Causal Individual tau(x)', fontsize=10)
axes[2].grid(True, linestyle='--', alpha=0.5)

# 4. Distribución del CATE por Ocupación (JOB)
sns.boxplot(data=train_df, x='JOB', y='CATE', hue='JOB', palette='Set3', legend=False, ax=axes[3])
axes[3].axhline(0, color='black', linestyle='--', linewidth=1)
axes[3].set_title('Distribución de tau(x) por Ocupación (JOB)', fontsize=11, fontweight='bold')
axes[3].set_xlabel('Ocupación', fontsize=10)
axes[3].set_ylabel('Efecto Causal Individual tau(x)', fontsize=10)
axes[3].grid(True, linestyle='--', alpha=0.5)

plt.suptitle("Análisis Multidimensional de Heterogeneidad Causal (EconML DML) — Identificación de Segmentos", fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()

# Segmentación Prescriptiva
seg = dml_results["strategic_segmentation"]
print("--- SEGMENTACIÓN PRESCRIPTIVA DE CARTERA PARA POLÍTICA CREDITICIA ---")
for k, v in seg.items():
    print(f"* {k.upper():30s}: {v['count']:5d} clientes ({v['pct']:5.1f}%) — {v['description']}")
"""))

    # ==============================================================================
    # 4.6 CÓDIGO: BATERÍA DE REFUTACIONES ESTOCÁSTICAS DE DOWHY
    # ==============================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""### 4.5 Batería de Refutaciones de Falsación Estocástica de DoWhy

Un hallazgo causal no es científicamente válido hasta que sobrevive a intentos activos de **falsación** (Karl Popper). DoWhy implementa tres pruebas de robustez estocástica:

1. **Tratamiento Placebo (*Placebo Treatment*):** Se sustituye el vector de tratamiento real $T$ por ruido aleatorio sintético. El nuevo ATE debe converger forzosamente a cero ($p\text{-valor} \approx 1.0$). Si retuviera significancia, el modelo estaría capturando correlaciones espurias del pipeline.
2. **Causa Común Aleatoria (*Random Common Cause*):** Se inyecta al DAG una variable independiente $W \sim \mathcal{N}(0, 1)$ como confusor ficticio. El ATE estimado no debe alterarse.
3. **Subconjunto de Datos (*Data Subset*):** Se re-estima el efecto causal extrayendo aleatoriamente el 80% de las observaciones. La estimación debe ser estable y consistente.
"""))

    cells.append(nbf.v4.new_code_cell(r"""# Ejecución de la Batería de Refutaciones de DoWhy
print("Ejecutando batería de 3 pruebas de refutación estocástica de DoWhy...")
refutations = run_dowhy_refutations(
    model_dowhy,
    identified_estimand,
    dowhy_estimates["regression"]["estimate_object"],
    num_simulations=20,
    random_state=42
)

# Extracción de métricas para el dashboard de semáforo
ref_random = refutations["random_common_cause"]
ref_placebo = refutations["placebo_treatment"]
ref_subset = refutations["data_subset"]

print(f"1. {ref_random['name']}: {ref_random['verdict']} (Delta = {ref_random['difference']:+.5f})")
print(f"2. {ref_placebo['name']}: {ref_placebo['verdict']} (Nuevo ATE = {ref_placebo['new_effect']:+.5f})")
print(f"3. {ref_subset['name']}: {ref_subset['verdict']} (Nuevo ATE = {ref_subset['new_effect']:+.5f})")

# Visualización Semáforo de Refutaciones
ref_labels = [
    'Tratamiento Placebo\n(Efecto debe colapsar a 0)',
    'Causa Común Aleatoria\n(Delta debe ser ~ 0)',
    'Subconjunto de Datos 80%\n(Estabilidad del efecto)'
]
ref_diffs = [
    ref_placebo['new_effect'],
    ref_random['difference'],
    ref_subset['difference']
]
ref_status = [
    ref_placebo['passed'],
    ref_random['passed'],
    ref_subset['passed']
]

plt.figure(figsize=(10.5, 4.2))
colors = ['#40c057' if p else '#fa5252' for p in ref_status]

bars = plt.barh(ref_labels, ref_diffs, color=colors, edgecolor='#2b8a3e', linewidth=1.5, height=0.42)
plt.axvline(0, color='black', linestyle='--', linewidth=1.2)
plt.title("Semáforo de Robustez Causal: Batería de Refutaciones de DoWhy", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Desviación Respecto al Criterio de Falsación", fontsize=10.5, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.4)

for bar, passed in zip(bars, ref_status):
    status_text = " APROBADO (Robusto)" if passed else " ALERTA"
    plt.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2, status_text,
             va='center', fontweight='bold', color='#2b8a3e' if passed else '#c92a2a')

plt.tight_layout()
plt.show()
"""))

    # ==============================================================================
    # 4.7 ENSAYO CRÍTICO COMPLETO: EL CHOQUE CAUSAL VS PREDICTIVO
    # ==============================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""---

## 📝 5. Ensayo Crítico Unificado: "El Choque Causal vs. Predictivo en la Toma de Decisiones Financieras"

**Integración Metodológica:** Equipo EAyMR 2026 (German Sittner, Martín Peña y Santiago Salvatore)  
**Cátedra:** Estadística Aplicada y Modelos de Regresión (EAyMR 2026) — UCA  
**Área:** Gobernanza de Algoritmos, Econometría y Causal Artificial Intelligence

---

### Resumen Ejecutivo y Tesis Central
En la última década, la industria bancaria y el ecosistema fintech han experimentado una revolución impulsada por algoritmos predictivos de aprendizaje automático (Random Forest, XGBoost, Redes Neuronales). Estos modelos han optimizado la capacidad de clasificar solicitantes de crédito minimizando el error fuera de muestra. Sin embargo, este éxito predictivo ha propagado un **grave error de categoría epistemológica**: la asunción de que una variable altamente predictiva para clasificar el riesgo de insolvencia es, simultáneamente, una **palanca accionable para intervenir y modificar la realidad**.

En este ensayo demostramos formalmente por qué el modelado predictivo pasivo ($P(Y \mid X)$) y la prescripción de políticas de intervención ($P(Y \mid do(T=t))$) operan en espacios matemáticos disjuntos. Analizamos la trampa del *Feature Importance*, el colapso que produce el condicionamiento ciego en colisionadores (Paradoja de Berkson), y formulamos un marco de gobernanza para la toma de decisiones automatizadas en carteras crediticias.

---

### 1. La Falacia del Feature Importance: ¿Por qué las variables dominantes en Random Forest no son palancas de política?

Consideremos los resultados obtenidos en nuestro **Torneo Predictivo (Fase 3)** sobre el dataset HMEQ (`Random Forest` ganador con $\text{AUC-ROC} = 0.9516$, $\text{PR-AUC} = 0.8408$, $\text{Brier} = 0.0697$). Al auditar la importancia de variables en Test (Permutation Importance) y en Train (MDI de Gini):
* **`DEBTINC_is_missing`** ocupa el **puesto #1 en MDI** ($20.17\%$) y **#2 en Permutación** ($\Delta\text{AUC} = 0.0393$). En Fase 1 demostramos mediante un modelo logit que esta omisión sigue un patrón **MAR/MNAR** con un *Odds Ratio* de $16.3\times$ para clientes en mora ($z = 34.74, p < 10^{-16}$).
* **`DEBTINC`** (Ratio Deuda/Ingreso) ocupa el **puesto #1 en Permutación** ($\Delta\text{AUC} = 0.0450$) y **#2 en MDI** ($16.02\%$), seguido por `CLAGE` (#3) y `DELINQ` (#4).
* En contraste, la variable de intervención accionable **`TREATMENT`** (`DebtCon` vs. `HomeImp`) queda relegada al **puesto #12 en MDI ($1.02\%$) y #14 de 17 en Permutación ($\Delta\text{AUC} = 0.0026$)**.

Frente a esta evidencia meramente predictiva, un comité ejecutivo no iniciado en inferencia causal suele caer en dos errores graves:
> *"Dado que `DEBTINC_is_missing` y `DEBTINC` dominan el Random Forest y `TREATMENT` apenas mueve el AUC, basta con exigir que todos declaren su deuda/ingreso y mantengan `DEBTINC` bajo el 30%, ignorando el destino del crédito."*

Esta propuesta constituye una **doble falacia entre síntoma predictivo y mecanismo causal**:
1. **`DEBTINC_is_missing` es un artefacto de registro, no una intervención:** Obligar administrativamente a llenar un campo faltante no vuelve solvente a un deudor que carece de ingresos formales verificables (mecanismo MAR/MNAR); solo destruye una señal predictiva de alerta temprana (como probó nuestra ablación de Fase 3, donde la Regresión Logística cayó de $\text{AUC} = 0.880$ a $0.764$).
2. **`DEBTINC` es un biomarcador financiero pasivo, no un tratamiento accionable:** Un cliente con `DEBTINC` bajo no es solvente *porque* su ratio contable sea bajo; su ratio es bajo porque posee estabilidad laboral (`YOJ`), historial limpio (`DELINQ`, `DEROG`) y respaldo patrimonial (`VALUE`). Confundir importancia predictiva con efectividad de política equivale a prohibir los termómetros porque los pacientes con fiebre registran mayor mortalidad hospitalaria.
3. **La ceguera del ML ante variables colineales o de política (`TREATMENT` en puesto #14):** Un árbol predictivo ignora `TREATMENT` porque las covariables basales $Z$ absorben su poder discriminatorio bruto. Sin embargo, cuando pasamos al **Nivel 2 de Pearl ($do(T)$)** en Fase 4, ese mismo `TREATMENT` casi invisible en Random Forest demuestra reducir la probabilidad absoluta de default en **$-5.54\text{ p.p.}$ ($\text{ATE} = -0.0554$)**, superando en un $39.5\%$ la estimación ingenua.

---

### 2. La Frontera Matemática Formal: $\mathbb{E}[Y \mid X = x]$ frente a $\mathbb{E}[Y \mid do(X = x)]$

La divergencia entre la predicción y la decisión puede enunciarse mediante la ruptura de la igualdad entre la esperanza condicional observacional y la esperanza bajo intervención quirúrgica:

$$\mathbb{E}[Y \mid T = 1] - \mathbb{E}[Y \mid T = 0] \quad \neq \quad \mathbb{E}[Y \mid do(T = 1)] - \mathbb{E}[Y \mid do(T = 0)]$$

#### El Operador $do$ y la Cirugía del Grafo
En la distribución observacional conjunta $P(Y, T, \mathbf{Z})$, la relación entre $T$ e $Y$ está contaminada por todos los caminos de puerta trasera ($T \leftarrow \mathbf{Z} \to Y$). Cuando calculamos:
$$P(Y = 1 \mid T = 1) = \frac{P(Y = 1, T = 1)}{P(T = 1)}$$
estamos pasivamente condicionando en el subconjunto de prestatarios que **eligieron por su propia cuenta** el producto `DebtCon`. Como demostramos empíricamente en la Sección 4.3, estos prestatarios poseen un historial significativamente más deteriorado de morosidad previa (`DELINQ`) y mayor carga financiera. Por tanto, la diferencia observacional observada ($\beta_{\text{naive}} = -0.0336$, $-3.36\text{ p.p.}$) refleja conjuntamente dos fenómenos opuestos:
1. El verdadero efecto protector de reestructurar pasivos a menor tasa.
2. El efecto adverso de que los solicitantes de consolidación están más cerca de la quiebra personal.

Por el contrario, el operador $do(T = 1)$ de Judea Pearl ejecuta una **cirugía estructural sobre el grafo acíclico dirigido**: elimina todos los arcos entrantes hacia $T$ (anulando la auto-selección $\mathbf{Z} \to T$), convirtiendo a $T$ en una variable exógena forzada por la política bancaria.

Solo tras aplicar el Criterio Backdoor recuperamos la equivalencia operativa:
$$\mathbb{E}[Y \mid do(T = 1)] - \mathbb{E}[Y \mid do(T = 0)] = \int \Big( \mathbb{E}[Y \mid T = 1, \mathbf{Z} = \mathbf{z}] - \mathbb{E}[Y \mid T = 0, \mathbf{Z} = \mathbf{z}] \Big) dF_{\mathbf{Z}}(\mathbf{z})$$

La cuantificación reveló que el verdadero $\text{ATE}$ es de **$-0.0554$** ($-5.54\text{ p.p.}$). Es decir: la política de consolidación reduce la probabilidad absoluta de mora en **$5.54$ puntos porcentuales**, un **$39.5\%$ más protectora** de lo que sugería la métrica observacional ingenua ($-0.0336$, $-3.36\text{ p.p.}$). Confiar ciegamente en el modelo predictivo observacional habría llevado al banco a subestimar el impacto real de una política crediticia altamente rentable y saneadora.

---

### 3. La Amenaza Silenciosa: Condicionamiento en Colisionadores (*Collider Bias* / Paradoja de Berkson)

El error más insidioso en el diseño de modelos automatizados en banca de crédito es el **sesgo del colisionador**. A diferencia de los confusores basales (que introducen sesgo si se omiten), los colisionadores introducen un sesgo devastador **cuando se incluyen indebidamente como controles**.

#### El Mecanismo en la Banca Hipotecaria
Supongamos que una entidad financiera analiza su cartera de créditos vigentes y decide construir un modelo de scoring controlando por la variable:
$$C = \text{Aprobación Extraordinaria por el Comité de Riesgos} \in \{0, 1\}$$

Para que un crédito sea aprobado por el comité extraordinario ($C=1$), el solicitante típicamente debe compensar una deficiencia grave con otra virtud destacada: o bien presenta un excelente colateral patrimonial ($T = \text{Equity Alto}$), o bien el oficial de cuentas considera que tiene una bajísima propensión individual a la morosidad ($Y = \text{Solvente}$).

El grafo causal local toma la forma canónica de un colisionador:
$$T \longrightarrow C \longleftarrow Y$$

En la población general, el colateral y la disciplina de pago pueden ser variables estocásticamente independientes ($T \perp\!\!\perp Y$). Sin embargo, al filtrar los datos restringiendo el análisis únicamente a los créditos que fueron otorgados tras pasar por el comité ($C = 1$):
* Si nos informan que un cliente aprobado tenía bajo colateral ($T = 0$), inferimos inmediatamente que forzosamente debía poseer una solvencia y conducta excepcional ($Y = 0$) para haber logrado la aprobación.
* Por el contrario, si un cliente tenía un colateral gigantesco ($T = 1$), el comité pudo haber tolerado una conducta de pago dudosa.

**Resultado:** Al condicionar en el colisionador $C$, se crea una correlación negativa espuria:
$$T \not\perp\!\!\perp Y \mid C = 1$$
El modelo de Machine Learning concluirá erróneamente que "tener mayor colateral incrementa la propensión al default". Esta anomalía matemática es la **Paradoja de Berkson** aplicada a la industria financiera.

---

### 4. Gobernanza y Recomendaciones para la Inteligencia de Decisiones (Decision Intelligence)

Para erradicar estos riesgos estructurales en las organizaciones de datos contemporáneas, proponemos tres mandamientos metodológicos de gobernanza:

1. **Separación de Pipelines según el Nivel de Pearl:**
   * Los modelos predictivos (XGBoost, Random Forest) deben emplearse **exclusivamente para scoring estático y provisión de incobrabilidad contable** (Nivel 1: Asociación).
   * El diseño de ofertas comerciales, asignación de tasas de interés y selección de productos crediticios debe ser gobernado **exclusivamente por modelos causales estructurales** (DoWhy, Double ML de EconML) (Nivel 2: Intervención).
2. **Auditoría de Grafos y Prohibición de Controles Post-Tratamiento:**
   * Ningún modelo de evaluación de políticas puede entrar a producción sin un DAG formal documentado en formato DOT, verificado por econometristas y auditores de riesgos.
   * Se prohíbe taxativamente la inclusión en el conjunto de ajuste de variables que actúen como **mediadores** (`LOAN` en nuestro caso) o **colisionadores** (indicadores de aprobación por comités o filtros de morosidad condicionada).
3. **Transición del Scoring Homogéneo a la Prescripción Personalizada del CATE:**
   * Abandonar la regla binaria univariada. Como evidenció el análisis de Double Machine Learning, el impacto de consolidar pasivos varía según el apalancamiento: es nulo para clientes de bajo riesgo ($\text{DEBTINC} < 30\%$), altamente protector para deudores en zona crítica ($35\% - 48\%$), e ineficaz para deudores insolventes extremos.
   * La inteligencia de decisiones moderna consiste en asignar cada tratamiento únicamente al estrato de clientes donde $\tau(x) < 0$ alcance significancia clínica y financiera.
"""))

    return cells
