"""
app/app.py
Fase 5 — Web App interactiva (Streamlit) del TP Integrador EAyMR 2026.

Tres módulos, uno por pestaña:
  1. Diagnóstico y exploración (filtros, variables, proyección PCA 2D).
  2. Predictivo "What-Is" (Random Forest de la Fase 3).
  3. Causal "What-If" (DAG, ATE, CATE y refutaciones de la Fase 4).

La app no recalcula nada pesado: lee los artefactos de artifacts/ que generan los
módulos de src/. Para correrla en local:  streamlit run app/app.py
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st
from scipy.interpolate import RegularGridInterpolator
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
ART = RAIZ / "artifacts"
DATOS = RAIZ / "data" / "processed"

# Paleta: dos tonos fijos para el resultado y uno más para resaltar
AZUL, NARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
GRIS, GRIS_CLARO, TINTA = "#8a8985", "#d9d8d3", "#1c1c1b"

NUMERICAS = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
OCUPACIONES = ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"]
NOMBRES = {
    "LOAN": "Monto del préstamo (USD)",
    "MORTDUE": "Hipoteca adeudada (USD)",
    "VALUE": "Valor de la propiedad (USD)",
    "YOJ": "Años en el empleo actual",
    "DEROG": "Reportes derogatorios",
    "DELINQ": "Líneas de crédito en mora",
    "CLAGE": "Antigüedad de la línea más vieja (meses)",
    "NINQ": "Consultas crediticias recientes",
    "CLNO": "Cantidad de líneas de crédito",
    "DEBTINC": "Ratio deuda / ingreso (%)",
    "JOB": "Ocupación",
    "REASON": "Propósito del crédito",
}
PROPOSITOS = {"DebtCon": "Consolidación de deuda", "HomeImp": "Mejoras del hogar"}

st.set_page_config(page_title="Riesgo crediticio HMEQ", page_icon="🏦", layout="wide")


# ----------------------------------------------------------------------------- utilidades
def pct(x, dec=1):
    return f"{100 * x:.{dec}f} %".replace(".", ",")


def pp(x, dec=1):
    """Puntos porcentuales con signo."""
    return f"{100 * x:+.{dec}f} pp".replace(".", ",")


def miles(x):
    return f"{x:,.0f}".replace(",", ".")


def wilson(k, n, z=1.96):
    """Intervalo de Wilson al 95 % para una proporción."""
    p = k / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    radio = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - radio, centro + radio


def estilo(fig, alto=380, leyenda=True):
    fig.update_layout(
        height=alto, margin=dict(l=10, r=10, t=30, b=10), separators=",.",
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color=TINTA, size=13),
        showlegend=leyenda, legend=dict(orientation="h", y=1.08, x=0),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(gridcolor="#eeeeea", zerolinecolor=GRIS_CLARO, linecolor=GRIS_CLARO)
    fig.update_yaxes(gridcolor="#eeeeea", zerolinecolor=GRIS_CLARO, linecolor=GRIS_CLARO)
    return fig


# ----------------------------------------------------------------------------- carga (una sola vez)
@st.cache_data
def cargar_datos():
    train = pd.read_csv(DATOS / "train.csv").assign(Conjunto="Train")
    test = pd.read_csv(DATOS / "test.csv").assign(Conjunto="Test")
    return train, test


@st.cache_data
def cargar_json(nombre):
    return json.loads((ART / nombre).read_text(encoding="utf-8"))


def _preprocesador():
    return ColumnTransformer([
        ("num_bin", "passthrough", NUMERICAS + ["DEBTINC_is_missing", "TREATMENT"]),
        ("job_ohe", OneHotEncoder(categories=[OCUPACIONES], drop="first", dtype=int,
                                  handle_unknown="ignore", sparse_output=False), ["JOB"]),
    ])


@st.cache_resource(show_spinner="Cargando el modelo predictivo…")
def cargar_modelos():
    """Random Forest ganador de la Fase 3. Si el archivo guardado es de otra versión de
    scikit-learn, se reentrena con los mismos hiperparámetros y la misma semilla."""
    train, _ = cargar_datos()
    bosque = None
    try:
        candidato = joblib.load(ART / "best_classifier.joblib")
        version = candidato.named_steps["classifier"].__getstate__().get("_sklearn_version")
        if version == sklearn.__version__:
            bosque = candidato
    except Exception:
        bosque = None
    if bosque is None:
        params = cargar_json("ml_metrics.json")["winner"]["best_params"]
        bosque = Pipeline([("preprocessor", _preprocesador()),
                           ("classifier", RandomForestClassifier(**params, random_state=42, n_jobs=-1))])
        bosque.fit(train, train["BAD"])
    # Logística Ridge (C = 0,01): la configuración que eligió la validación cruzada en la Fase 3
    logistica = Pipeline([("preprocessor", clone(bosque.named_steps["preprocessor"])),
                          ("escala", StandardScaler()),
                          ("modelo", LogisticRegression(C=0.01, max_iter=5000))]).fit(train, train["BAD"])
    return bosque, logistica


@st.cache_data
def predicciones_test():
    _, test = cargar_datos()
    bosque, _ = cargar_modelos()
    return bosque.predict_proba(test)[:, 1], test["BAD"].values


@st.cache_resource
def cargar_pca():
    """PCA de la Fase 2 (escalador y componentes ajustados solo con train)."""
    train, test = cargar_datos()
    try:
        b = joblib.load(ART / "pca_pipeline.joblib")
        cols, escalador, pca = b["feature_cols"], b["scaler"], b["pca"]
        escalador.transform(train[cols].head(1))
    except Exception:
        cols = [c for c in NUMERICAS if c != "LOAN"]
        escalador = StandardScaler().fit(train[cols])
        pca = PCA().fit(escalador.transform(train[cols]))
    cargas = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), index=cols,
                          columns=[f"PC{i + 1}" for i in range(len(cols))])
    return cols, escalador, pca, cargas


@st.cache_data
def base_completa():
    train, test = cargar_datos()
    cols, escalador, pca, _ = cargar_pca()
    df = pd.concat([train, test], ignore_index=True)
    z = pca.transform(escalador.transform(df[cols]))
    df["PC1"], df["PC2"] = z[:, 0], z[:, 1]
    df["Resultado"] = np.where(df["BAD"] == 1, "Default", "Cumple")
    df["Propósito"] = df["REASON"].map(PROPOSITOS)
    df["DEBTINC informado"] = np.where(df["DEBTINC_is_missing"] == 1, "No informado", "Informado")
    return df


@st.cache_resource
def cargar_cate():
    """CATE del CausalForestDML precalculado (src/export_app_cate.py)."""
    z = np.load(ART / "app_cate_grid.npz")
    variables = [str(v) for v in z["variables"]]
    ejes = [z[f"eje_{i}"] for i in range(len(variables))]
    interp = {k: RegularGridInterpolator(ejes, z[k], bounds_error=False, fill_value=None)
              for k in ("tau", "tau_inf", "tau_sup")}
    return variables, ejes, interp, z["tau_train"], z["tau_train_inf"], z["tau_train_sup"]


train, test = cargar_datos()
MEDIANAS = train[NUMERICAS].median()

# Perfil del simulador What-Is: se guarda en la sesión para poder proyectarlo en el PCA
DEFECTOS = {"wi_LOAN": 16000, "wi_MORTDUE": 65000, "wi_VALUE": 90000, "wi_YOJ": 7, "wi_DEROG": 0,
            "wi_DELINQ": 0, "wi_CLAGE": 174, "wi_NINQ": 1, "wi_CLNO": 20, "wi_DEBTINC": 34.8,
            "wi_sin_debtinc": False, "wi_JOB": "Other", "wi_REASON": "DebtCon"}
for clave, valor in DEFECTOS.items():
    st.session_state.setdefault(clave, valor)


def perfil_actual() -> pd.DataFrame:
    s = st.session_state
    sin_dato = bool(s["wi_sin_debtinc"])
    fila = {v: float(s[f"wi_{v}"]) for v in NUMERICAS}
    if sin_dato:  # igual que en el Sprint 0: se imputa la mediana de train y se marca el faltante
        fila["DEBTINC"] = float(MEDIANAS["DEBTINC"])
    fila.update(DEBTINC_is_missing=int(sin_dato), JOB=s["wi_JOB"], REASON=s["wi_REASON"],
                TREATMENT=int(s["wi_REASON"] == "DebtCon"))
    return pd.DataFrame([fila])


# ----------------------------------------------------------------------------- encabezado
st.title("Riesgo crediticio en préstamos con garantía hipotecaria")
st.caption("TP Integrador · Estadística Aplicada y Modelos de Regresión (UCA, 2026) · Dataset HMEQ: "
           "5.960 préstamos, 20 % en default")

tab_diag, tab_is, tab_if = st.tabs(["1 · Diagnóstico y exploración", "2 · Predictivo (What-Is)",
                                    "3 · Causal (What-If)"])

# ============================================================================= 1. DIAGNÓSTICO
with tab_diag:
    df = base_completa()
    st.subheader("Filtrar la muestra")
    f1, f2, f3, f4 = st.columns(4)
    conjuntos = f1.multiselect("Conjunto", ["Train", "Test"], default=["Train", "Test"])
    resultados = f2.multiselect("Resultado", ["Cumple", "Default"], default=["Cumple", "Default"])
    propositos = f3.multiselect("Propósito del crédito", list(PROPOSITOS.values()), default=list(PROPOSITOS.values()))
    informados = f4.multiselect("Ratio deuda / ingreso", ["Informado", "No informado"],
                                default=["Informado", "No informado"])
    f5, f6 = st.columns([2, 2])
    ocupaciones = f5.multiselect("Ocupación", OCUPACIONES, default=OCUPACIONES)
    lo, hi = int(df["LOAN"].min()), int(df["LOAN"].max())
    rango = f6.slider("Monto del préstamo (USD)", lo, hi, (lo, hi), step=100)

    sel = df[df["Conjunto"].isin(conjuntos) & df["Resultado"].isin(resultados) & df["Propósito"].isin(propositos)
             & df["DEBTINC informado"].isin(informados) & df["JOB"].isin(ocupaciones)
             & df["LOAN"].between(*rango)]

    if sel.empty:
        st.warning("Ningún cliente cumple con esos filtros. Ampliá la selección.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clientes seleccionados", miles(len(sel)), f"{pct(len(sel) / len(df))} de la muestra",
                  delta_color="off", delta_arrow="off")
        dif = sel["BAD"].mean() - df["BAD"].mean()
        k2.metric("Tasa de default", pct(sel["BAD"].mean()), f"{pp(dif)} vs. total",
                  delta_color="inverse" if abs(dif) >= 0.0005 else "off")
        k3.metric("Monto mediano", f"USD {miles(sel['LOAN'].median())}")
        k4.metric("Sin ratio deuda / ingreso", pct(sel["DEBTINC_is_missing"].mean()))

        st.divider()
        izq, der = st.columns([1, 1], gap="large")

        # ---- inspector de variables
        with izq:
            st.subheader("Inspeccionar una variable")
            var = st.selectbox("Variable", NUMERICAS + ["JOB", "REASON"], format_func=NOMBRES.get)
            fig = go.Figure()
            if var in NUMERICAS:
                tope = sel[var].quantile(0.99)
                bins = dict(start=float(sel[var].min()), end=float(tope), size=float(max(tope - sel[var].min(), 1) / 30))
                for nombre, color in (("Cumple", AZUL), ("Default", NARANJA)):
                    fig.add_histogram(x=sel.loc[sel["Resultado"] == nombre, var], name=nombre, marker_color=color,
                                      opacity=0.75, xbins=bins, histnorm="percent",
                                      hovertemplate="%{x}<br>%{y:.1f} % del grupo<extra>" + nombre + "</extra>")
                fig.update_layout(barmode="overlay", bargap=0.05)
                fig.update_xaxes(title=NOMBRES[var])
                fig.update_yaxes(title="% de clientes de cada grupo")
                st.plotly_chart(estilo(fig, 340), width="stretch")
                st.caption("Cada grupo suma 100 %. El eje se corta en el percentil 99 para que los extremos "
                           "no aplasten el gráfico.")
                resumen = sel.groupby("Resultado")[var].describe()[["count", "mean", "50%", "std", "min", "max"]]
                resumen.columns = ["Clientes", "Media", "Mediana", "Desvío", "Mínimo", "Máximo"]
                st.dataframe(resumen.style.format(lambda v: f"{v:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")),
                             width="stretch")
            else:
                tasa = sel.groupby(var)["BAD"].agg(["mean", "size"]).sort_values("mean")
                etiquetas = [PROPOSITOS.get(i, i) for i in tasa.index]
                fig.add_bar(y=etiquetas, x=tasa["mean"], orientation="h", marker_color=AZUL, customdata=tasa["size"],
                            text=[pct(v) for v in tasa["mean"]], textposition="outside", cliponaxis=False,
                            hovertemplate="%{y}<br>Default: %{x:.1%}<br>Clientes: %{customdata}<extra></extra>")
                fig.add_vline(x=sel["BAD"].mean(), line_dash="dot", line_color=GRIS,
                              annotation_text="promedio", annotation_font_color=GRIS)
                fig.update_xaxes(title="Tasa de default", tickformat=".0%", range=[0, max(tasa["mean"]) * 1.25])
                st.plotly_chart(estilo(fig, 340, leyenda=False), width="stretch")
                tabla = tasa.rename(columns={"mean": "Tasa de default", "size": "Clientes"})
                tabla.index = etiquetas
                st.dataframe(tabla.style.format({"Tasa de default": pct}), width="stretch")

        # ---- PCA
        with der:
            cols_pca, escalador, pca, cargas = cargar_pca()
            var_exp = pca.explained_variance_ratio_
            st.subheader("Proyección en las dos primeras componentes principales")
            c1, c2 = st.columns([2, 2])
            color_por = c1.selectbox("Colorear por", ["Resultado", "Propósito", "DEBTINC informado"] + NUMERICAS,
                                     format_func=lambda v: NOMBRES.get(v, v))
            ver_cliente = c2.checkbox("Mostrar el cliente del simulador What-Is", value=True)

            fig = go.Figure()
            hover = ("PC1: %{x:.2f} · PC2: %{y:.2f}<br>Préstamo: USD %{customdata[0]:,.0f}<br>"
                     "Deuda/ingreso: %{customdata[1]:.1f}<br>Moras: %{customdata[2]:.0f}")
            datos_hover = ["LOAN", "DEBTINC", "DELINQ"]
            if color_por in NUMERICAS:
                tope = sel[color_por].quantile(0.99)
                fig.add_scattergl(x=sel["PC1"], y=sel["PC2"], mode="markers", customdata=sel[datos_hover],
                                  marker=dict(size=5, color=sel[color_por].clip(upper=tope), colorscale="Blues",
                                              cmin=float(sel[color_por].min()), cmax=float(tope), opacity=0.8,
                                              colorbar=dict(title=color_por, thickness=12)),
                                  hovertemplate=hover + "<extra></extra>", showlegend=False)
            else:
                for (nombre, grupo), color in zip(sel.groupby(color_por, sort=True), (AZUL, NARANJA, VERDE)):
                    fig.add_scattergl(x=grupo["PC1"], y=grupo["PC2"], mode="markers", name=str(nombre),
                                      customdata=grupo[datos_hover], marker=dict(size=5, color=color, opacity=0.55),
                                      hovertemplate=hover + "<extra>" + str(nombre) + "</extra>")
            if ver_cliente:
                zc = pca.transform(escalador.transform(perfil_actual()[cols_pca]))[0]
                fig.add_scattergl(x=[zc[0]], y=[zc[1]], mode="markers", name="Cliente What-Is",
                                marker=dict(symbol="star", size=20, color="#111111", line=dict(color="white", width=2)),
                                hovertemplate="Cliente del simulador<br>PC1: %{x:.2f} · PC2: %{y:.2f}<extra></extra>")
            fig.update_xaxes(title=f"PC1 ({pct(var_exp[0])} de la varianza)", range=[-4, 8])
            fig.update_yaxes(title=f"PC2 ({pct(var_exp[1])} de la varianza)", range=[-4, 8])
            st.plotly_chart(estilo(fig, 470), width="stretch")
            st.caption(f"Las dos componentes juntas explican el {pct(var_exp[0] + var_exp[1])} de la varianza de "
                       "las 9 variables numéricas (sin el monto del préstamo). El escalado y las componentes se "
                       "ajustaron solo con train. La vista se limita al rango −4 a 8; algunos casos extremos quedan fuera.")

            with st.expander("¿Qué representa cada componente?"):
                top = cargas[["PC1", "PC2"]].copy()
                orden = top.abs().max(axis=1).sort_values().index
                figc = go.Figure()
                for comp, color in (("PC1", AZUL), ("PC2", NARANJA)):
                    figc.add_bar(y=[NOMBRES[v].split(" (")[0] for v in orden], x=top.loc[orden, comp], name=comp,
                                 orientation="h", marker_color=color,
                                 hovertemplate="%{y}<br>Carga en " + comp + ": %{x:.2f}<extra></extra>")
                figc.update_xaxes(title="Carga (correlación de la variable con la componente)", range=[-1, 1])
                st.plotly_chart(estilo(figc, 360), width="stretch")
                st.markdown(
                    "- **PC1** se mueve con la hipoteca adeudada, el valor de la propiedad y la cantidad de líneas "
                    "de crédito: resume el **tamaño patrimonial** del cliente.\n"
                    "- **PC2** sube con los reportes derogatorios, las consultas recientes, el ratio deuda/ingreso "
                    "y las moras, y baja con la antigüedad crediticia y laboral: resume el **estrés crediticio**.")

# ============================================================================= 2. WHAT-IS
with tab_is:
    bosque, logistica = cargar_modelos()
    metricas = cargar_json("ml_metrics.json")
    st.subheader("Simulador: ¿qué probabilidad de default tiene este solicitante?")
    st.caption(f"Modelo: Random Forest ganador del torneo de la Fase 3 (AUC en test = "
               f"{metricas['winner']['test_roc_auc']:.3f}".replace(".", ",") + "). Completá el perfil y la "
               "predicción se actualiza sola.")

    entrada, salida = st.columns([5, 6], gap="large")
    with entrada:
        st.markdown("**Préstamo y propiedad**")
        a, b = st.columns(2)
        a.number_input(NOMBRES["LOAN"], 1000, 90000, step=500, key="wi_LOAN")
        b.selectbox(NOMBRES["REASON"], list(PROPOSITOS), format_func=PROPOSITOS.get, key="wi_REASON")
        a.number_input(NOMBRES["MORTDUE"], 0, 400000, step=1000, key="wi_MORTDUE")
        b.number_input(NOMBRES["VALUE"], 8000, 860000, step=1000, key="wi_VALUE")

        st.markdown("**Situación laboral e ingresos**")
        a, b = st.columns(2)
        a.selectbox(NOMBRES["JOB"], OCUPACIONES, key="wi_JOB")
        b.number_input(NOMBRES["YOJ"], 0, 41, key="wi_YOJ")
        a.number_input(NOMBRES["DEBTINC"], 0.0, 200.0, step=0.5, key="wi_DEBTINC",
                       disabled=st.session_state["wi_sin_debtinc"])
        b.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        b.checkbox("No informó su ratio deuda / ingreso", key="wi_sin_debtinc")

        st.markdown("**Historial crediticio**")
        a, b = st.columns(2)
        a.number_input(NOMBRES["DELINQ"], 0, 15, key="wi_DELINQ")
        b.number_input(NOMBRES["DEROG"], 0, 10, key="wi_DEROG")
        a.number_input(NOMBRES["NINQ"], 0, 17, key="wi_NINQ")
        b.number_input(NOMBRES["CLNO"], 0, 71, key="wi_CLNO")
        a.number_input(NOMBRES["CLAGE"], 0, 1200, key="wi_CLAGE")

    perfil = perfil_actual()
    p = float(bosque.predict_proba(perfil)[0, 1])
    p_logit = float(logistica.predict_proba(perfil)[0, 1])
    tasa_base = float(train["BAD"].mean())

    with salida:
        m1, m2 = st.columns(2)
        m1.metric("Probabilidad de default (Random Forest)", pct(p),
                  f"{p / tasa_base:.1f} veces la tasa base".replace(".", ","), delta_color="off", delta_arrow="off")
        m2.metric("Segunda opinión (Regresión Logística)", pct(p_logit),
                  f"{pp(p_logit - p)} vs. Random Forest", delta_color="off", delta_arrow="off")

        barra = go.Figure()
        barra.add_bar(x=[p], y=[""], orientation="h", marker_color=NARANJA if p >= tasa_base else AZUL,
                      width=0.5, hovertemplate="Probabilidad: %{x:.1%}<extra></extra>")
        barra.add_vline(x=tasa_base, line_dash="dot", line_color=TINTA,
                        annotation_text=f"tasa base {pct(tasa_base, 0)}", annotation_position="top")
        barra.update_xaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(estilo(barra, 110, leyenda=False), width="stretch")

        # ---- probabilidad calibrada: qué pasó en test con clientes de probabilidad parecida
        p_test, y_test = predicciones_test()
        k = 150
        vecinos = np.argsort(np.abs(p_test - p))[:k]
        observada = y_test[vecinos].mean()
        inf, sup = wilson(y_test[vecinos].sum(), k)
        st.markdown("**Probabilidad calibrada**")
        st.info(f"Entre los {k} clientes de test a los que el modelo les asignó una probabilidad más parecida "
                f"(de {pct(p_test[vecinos].min())} a {pct(p_test[vecinos].max())}), incumplió el "
                f"**{pct(observada)}**. Intervalo de confianza al 95 %: **{pct(inf)} a {pct(sup)}**.")
        st.caption("El intervalo es el de Wilson para una proporción. Sirve para chequear que la probabilidad "
                   "del modelo coincide con lo que pasó en datos que no vio.")

        # ---- explicación local
        st.markdown("**¿Qué variables explican esta predicción?**")
        ref = {**MEDIANAS.to_dict(), "JOB": train["JOB"].mode()[0], "REASON": train["REASON"].mode()[0]}
        grupos = {v: [v] for v in NUMERICAS if v != "DEBTINC"}
        grupos.update(DEBTINC=["DEBTINC", "DEBTINC_is_missing"], JOB=["JOB"], REASON=["REASON", "TREATMENT"])
        variantes = []
        for g in grupos:
            alt = perfil.copy()
            if g == "DEBTINC":
                alt[["DEBTINC", "DEBTINC_is_missing"]] = [ref["DEBTINC"], 0]
            elif g == "REASON":
                alt[["REASON", "TREATMENT"]] = [ref["REASON"], int(ref["REASON"] == "DebtCon")]
            else:
                alt[g] = ref[g]
            variantes.append(alt)
        p_alt = bosque.predict_proba(pd.concat(variantes, ignore_index=True))[:, 1]
        aporte = pd.Series(p - p_alt, index=list(grupos)).loc[lambda s: s.abs() >= 0.0005]
        aporte = aporte.reindex(aporte.abs().sort_values().index)

        def valor_mostrado(g):
            if g == "DEBTINC":
                return "no informado" if perfil["DEBTINC_is_missing"][0] else f"{perfil['DEBTINC'][0]:.1f}"
            if g == "REASON":
                return PROPOSITOS[perfil["REASON"][0]]
            return perfil[g][0] if g == "JOB" else miles(perfil[g][0])

        if aporte.empty:
            st.write("Este perfil coincide con el cliente típico: ninguna variable mueve la predicción.")
        else:
            figx = go.Figure()
            figx.add_bar(y=[f"{NOMBRES[g].split(' (')[0]} = {valor_mostrado(g)}" for g in aporte.index], x=aporte.values,
                         orientation="h", marker_color=[NARANJA if v > 0 else AZUL for v in aporte.values],
                         text=[pp(v) for v in aporte.values], textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}<br>Cambio en la probabilidad: %{x:+.1%}<extra></extra>")
            margen = max(aporte.abs().max() * 1.35, 0.02)
            figx.update_xaxes(range=[-margen, margen], tickformat="+.0%", title="Cambio en la probabilidad de default")
            st.plotly_chart(estilo(figx, 90 + 34 * len(aporte), leyenda=False), width="stretch")
        st.caption("Cada barra muestra cuánto cambia la probabilidad por tener ese valor en lugar del valor típico "
                   "(la mediana de train, o la categoría más frecuente). Naranja: sube el riesgo. Azul: lo baja. "
                   "Se cambia una variable por vez, así que las barras no tienen por qué sumar la diferencia total.")

        with st.expander("Importancia global de las variables en el modelo"):
            imp = pd.DataFrame(metricas["feature_importance"]).set_index("Feature")["Permutacion"].sort_values().tail(8)
            figi = go.Figure()
            figi.add_bar(y=imp.index, x=imp.values, orientation="h", marker_color=AZUL,
                         hovertemplate="%{y}<br>Caída del AUC: %{x:.3f}<extra></extra>")
            figi.update_xaxes(title="Importancia por permutación (caída del AUC en test)")
            st.plotly_chart(estilo(figi, 300, leyenda=False), width="stretch")
            st.caption("Esto mide asociación, no causa: que una variable prediga bien el default no quiere decir "
                       "que cambiarla lo reduzca. Esa pregunta la responde la pestaña What-If.")

# ============================================================================= 3. WHAT-IF
with tab_if:
    causal = cargar_json("causal_summary.json")
    ate = causal["ate_estimates"]
    ref_c = causal["refutations"]
    ate_reg = ate["backdoor_linear_regression"]
    variables, ejes, interp, tau_tr, tau_tr_inf, tau_tr_sup = cargar_cate()

    st.subheader("Simulación de política: ¿qué pasa si orientamos el crédito a consolidar deuda?")
    st.markdown("**Intervención evaluada:** `do(TREATMENT = 1)`, es decir, otorgar el crédito para "
                "**consolidación de deuda** en lugar de **mejoras del hogar**. **Resultado:** probabilidad de default.")

    # ---- semáforo: se arma con los valores de las tres refutaciones
    # IC 95 % del ATE original: misma regresión backdoor (BAD ~ T + confusores) por MCO
    confusores_num = [c for c in causal["dag_metadata"]["backdoor_adjustment_set"] if c != "JOB"]
    X_ate = np.column_stack([np.ones(len(train)), train["TREATMENT"], train[confusores_num],
                             pd.get_dummies(train["JOB"], drop_first=True).astype(float)])
    y_ate = train["BAD"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X_ate, y_ate, rcond=None)[0]
    resid = y_ate - X_ate @ beta
    ee_ate = float(np.sqrt(resid @ resid / (len(y_ate) - X_ate.shape[1]) * np.linalg.inv(X_ate.T @ X_ate)[1, 1]))
    ic_ate = (ate_reg - 1.96 * ee_ate, ate_reg + 1.96 * ee_ate)

    def luz(nombre):
        """Verde: cumple el criterio de la prueba. Amarillo: pasa el criterio del equipo pero
        queda fuera del umbral estricto. Rojo: no pasa."""
        nuevo, paso = ref_c[nombre]["new_effect"], ref_c[nombre]["passed"]
        if not paso:
            return "rojo"
        if nombre == "placebo_treatment":
            return "verde" if abs(nuevo) < 0.005 else "amarillo"
        return "verde" if ic_ate[0] <= nuevo <= ic_ate[1] else "amarillo"

    PRUEBAS = {
        "random_common_cause": ("Causa común aleatoria",
                                "Se agrega un confusor inventado, de puro ruido. El efecto no debería cambiar."),
        "placebo_treatment": ("Tratamiento placebo",
                              "Se reemplaza el tratamiento por uno al azar. El efecto debería caer a cero."),
        "data_subset": ("Subconjunto de datos",
                        "Se vuelve a estimar con el 80 % de la muestra. El efecto debería mantenerse."),
    }
    luces = {n: luz(n) for n in PRUEBAS}
    ICONO = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}
    n_verdes = sum(v == "verde" for v in luces.values())
    if "rojo" in luces.values():
        general, caja = "Confianza causal BAJA", st.error
    elif n_verdes == 3:
        general, caja = "Confianza causal ALTA", st.success
    else:
        general, caja = "Confianza causal MODERADA", st.warning

    st.markdown("#### Efecto promedio de la política (ATE)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("ATE · regresión con ajuste backdoor", pp(ate_reg), "método principal", delta_color="off", delta_arrow="off")
    a2.metric("ATE · ponderación por propensión (IPW)", pp(ate["propensity_score_weighting_ipw"]),
              "segundo método", delta_color="off", delta_arrow="off")
    a3.metric("Comparación ingenua (sin ajustar)", pp(ate["naive_observational_beta"]),
              f"sesgo de confusión {pp(ate['confounding_bias'])}", delta_color="off", delta_arrow="off")
    a4.metric("E-value", f"{causal['sensitivity_e_value']['e_value_point']:.2f}".replace(".", ","),
              "sensibilidad a confusores ocultos", delta_color="off", delta_arrow="off")
    st.markdown(
        f"Consolidar deuda **reduce la probabilidad de default en {pp(abs(ate_reg)).lstrip('+').replace(' pp', '')} puntos porcentuales** "
        f"en promedio (de {pct(causal['sensitivity_e_value']['risk_control_homeimp'])} a "
        f"{pct(causal['sensitivity_e_value']['risk_treated_debtcon'])}). La comparación ingenua mostraba menos "
        "efecto porque quienes piden consolidar deuda ya venían con un perfil más riesgoso. "
        f"Para anular este resultado haría falta un confusor no observado asociado al tratamiento y al default con "
        "un riesgo relativo de al menos " + f"{causal['sensitivity_e_value']['e_value_point']:.2f}".replace(".", ",") + ".")

    st.markdown("#### Semáforo de confianza: pruebas de refutación de DoWhy")
    caja(f"**{general}** · {n_verdes} de 3 pruebas en verde")
    tarjetas = st.columns(3)
    for col, (clave, (titulo, descripcion)) in zip(tarjetas, PRUEBAS.items()):
        nuevo = ref_c[clave]["new_effect"]
        with col.container(border=True):
            st.markdown(f"### {ICONO[luces[clave]]} {titulo}")
            st.caption(descripcion)
            st.markdown(f"Efecto original: **{pp(ate_reg, 2)}**  \nEfecto en la prueba: **{pp(nuevo, 2)}**")
            if clave == "placebo_treatment":
                st.caption("Verde: el efecto placebo queda a menos de 0,5 pp de cero.")
            else:
                st.caption(f"Verde: el efecto de la prueba queda dentro del intervalo de confianza al 95 % "
                           f"del efecto original ({pp(ic_ate[0], 2)} a {pp(ic_ate[1], 2)}).")

    st.divider()
    col_dag, col_cate = st.columns([1, 1], gap="large")

    # ---- DAG interactivo
    with col_dag:
        st.markdown("#### DAG causal")
        aristas = re.findall(r"(\w+)\s*->\s*(\w+)", (ART / "causal_dag.dot").read_text(encoding="utf-8"))
        confusores = sorted({a for a, b in aristas if b == "TREATMENT"})
        vista = st.radio("Resaltar", ["Todo", "Efecto causal", "Puertas traseras", "Mediador", "Colisionador"],
                         horizontal=True, label_visibility="collapsed")
        pos = {c: (i * 10 / (len(confusores) - 1), 3.0) for i, c in enumerate(confusores)}
        pos.update(TREATMENT=(1.5, 1.2), BAD=(8.5, 1.2), LOAN=(5, 0.1), COLLIDER_AUDIT=(5, -1.1))
        ROL = {"TREATMENT": ("Tratamiento (T)", AZUL, "1 = consolidación de deuda, 0 = mejoras del hogar"),
               "BAD": ("Resultado (Y)", NARANJA, "1 = default, 0 = cumple"),
               "LOAN": ("Mediador (M)", VERDE, "Monto del crédito: depende del tratamiento. No se ajusta por él."),
               "COLLIDER_AUDIT": ("Colisionador (C)", "#ffffff", "Auditoría posterior al crédito. Condicionar en "
                                  "él abriría una asociación falsa (paradoja de Berkson).")}

        def tipo(a, b):
            if (a, b) == ("TREATMENT", "BAD"):
                return "Efecto causal"
            if "COLLIDER_AUDIT" in (a, b):
                return "Colisionador"
            if "LOAN" in (a, b):
                return "Mediador"
            return "Puertas traseras"

        COLOR_ARISTA = {"Efecto causal": TINTA, "Puertas traseras": GRIS, "Mediador": VERDE, "Colisionador": GRIS}
        dag = go.Figure()
        for a, b in aristas:
            t = tipo(a, b)
            activa = vista in ("Todo", t)
            (x0, y0), (x1, y1) = pos[a], pos[b]
            dag.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowsize=1.2, standoff=17, startstandoff=14,
                               arrowwidth=(2.6 if t == "Efecto causal" else 1.3) if activa else 0.8,
                               arrowcolor=COLOR_ARISTA[t] if activa else "#e6e5e0", opacity=1 if activa else 0.7)
        nodos_activos = {n for a, b in aristas if vista in ("Todo", tipo(a, b)) for n in (a, b)}
        for n, (x, y) in pos.items():
            rol, color, ayuda = ROL.get(n, ("Confusor (Z)", "#f1f0ec", f"{NOMBRES.get(n, n)}. Afecta al "
                                           "tratamiento y al default: se ajusta por él."))
            activo = n in nodos_activos
            dag.add_scatter(x=[x], y=[y], mode="markers+text", text=["COLLIDER" if n == "COLLIDER_AUDIT" else n],
                            textposition="top center" if y > 2 else "bottom center",
                            textfont=dict(size=11, color=TINTA if activo else GRIS_CLARO),
                            marker=dict(size=26, color=color if activo else "#f6f6f3",
                                        line=dict(color=TINTA if activo else GRIS_CLARO, width=1.2)),
                            hovertemplate=f"<b>{n}</b><br>{rol}<br>{ayuda}<extra></extra>", showlegend=False)
        dag.update_xaxes(visible=False, range=[-0.8, 10.8])
        dag.update_yaxes(visible=False, range=[-1.9, 3.7])
        st.plotly_chart(estilo(dag, 430, leyenda=False), width="stretch")
        st.caption("Pasá el mouse por cada nodo para ver su rol. Azul: tratamiento. Naranja: resultado. Verde: "
                   "mediador. Gris: confusores. Blanco: colisionador. El conjunto de ajuste backdoor son los "
                   f"{len(causal['dag_metadata']['backdoor_adjustment_set'])} confusores; el mediador LOAN queda afuera.")

    # ---- CATE según el perfil
    with col_cate:
        st.markdown("#### Efecto según el perfil del cliente (CATE)")
        claves = {v: f"cate_{v}" for v in variables}
        limites = {v: (float(e[0]), float(e[-1])) for v, e in zip(variables, ejes)}
        for v in variables:
            st.session_state.setdefault(claves[v], float(np.clip(MEDIANAS[v], *limites[v])))

        def copiar_perfil():
            fila = perfil_actual().iloc[0]
            for v in variables:
                st.session_state[claves[v]] = float(np.clip(fila[v], *limites[v]))

        s1, s2, s3 = st.columns(3)
        s1.slider("Deuda / ingreso (%)", *[round(x, 1) for x in limites["DEBTINC"]], step=0.5, key=claves["DEBTINC"])
        s2.slider("Valor propiedad (USD)", *limites["VALUE"], step=1000.0, format="%.0f", key=claves["VALUE"])
        s3.slider("Antigüedad crediticia (meses)", *[round(x) * 1.0 for x in limites["CLAGE"]], step=1.0,
                  format="%.0f", key=claves["CLAGE"])
        b1, b2 = st.columns([3, 2])
        eje_x = b1.selectbox("Variable del eje horizontal", variables, format_func=lambda v: NOMBRES[v])
        b2.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        b2.button("Usar el cliente del What-Is", on_click=copiar_perfil, width="stretch")

        punto = np.array([[float(np.clip(st.session_state[claves[v]], *limites[v])) for v in variables]])
        i = variables.index(eje_x)
        curva = np.repeat(punto, 60, axis=0)
        curva[:, i] = np.linspace(*limites[eje_x], 60)
        t, ti, ts = (interp[k](curva) for k in ("tau", "tau_inf", "tau_sup"))
        t0, ti0, ts0 = (float(interp[k](punto)[0]) for k in ("tau", "tau_inf", "tau_sup"))

        figc = go.Figure()
        figc.add_scatter(x=curva[:, i], y=ts, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        figc.add_scatter(x=curva[:, i], y=ti, mode="lines", line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(42,120,214,0.15)", name="IC 95 %", hoverinfo="skip")
        figc.add_scatter(x=curva[:, i], y=t, mode="lines", line=dict(color=AZUL, width=2.5), name="CATE",
                         customdata=np.c_[ti, ts],
                         hovertemplate="%{x:,.1f}<br>Efecto: %{y:+.1%}<br>IC 95 %: %{customdata[0]:+.1%} a "
                                       "%{customdata[1]:+.1%}<extra></extra>")
        figc.add_scatter(x=[punto[0, i]], y=[t0], mode="markers", name="Perfil elegido",
                         marker=dict(size=13, color=TINTA, line=dict(color="white", width=2)),
                         hovertemplate="Perfil elegido<br>Efecto: %{y:+.1%}<extra></extra>")
        figc.add_hline(y=0, line_color=TINTA, line_width=1)
        figc.add_hline(y=ate_reg, line_dash="dot", line_color=GRIS, annotation_text="ATE", annotation_font_color=GRIS)
        figc.update_xaxes(title=NOMBRES[eje_x])
        figc.update_yaxes(title="Efecto sobre la prob. de default", tickformat="+.0%")
        st.plotly_chart(estilo(figc, 340), width="stretch")

        if ts0 < 0:
            st.success(f"Para este perfil, consolidar deuda **reduce** el default: {pp(t0)} "
                       f"(IC 95 %: {pp(ti0)} a {pp(ts0)}).")
        elif ti0 > 0:
            st.error(f"Para este perfil, consolidar deuda **aumenta** el default: {pp(t0)} "
                     f"(IC 95 %: {pp(ti0)} a {pp(ts0)}).")
        else:
            st.warning(f"Para este perfil el efecto estimado es {pp(t0)}, pero **no es concluyente**: el "
                       f"intervalo al 95 % ({pp(ti0)} a {pp(ts0)}) incluye el cero.")
        st.caption("Bosque causal (CausalForestDML) de la Fase 4, que deja variar el efecto según estas tres "
                   "variables. Valores negativos: la política baja el default. La curva mueve una variable y "
                   "deja las otras dos fijas en el perfil elegido.")

    # ---- política por segmento
    st.divider()
    st.markdown("#### ¿A qué clientes conviene aplicar la política?")
    st.caption("Elegí un segmento y la app resume el efecto estimado para los clientes de train que caen en él.")
    g1, g2, g3 = st.columns(3)
    r_debt = g1.slider("Deuda / ingreso (%)", 0.0, 60.0, (0.0, 60.0), step=1.0, key="seg_debt")
    r_val = g2.slider("Valor de la propiedad (miles de USD)", 0, 400, (0, 400), step=10, key="seg_val")
    r_age = g3.slider("Antigüedad crediticia (meses)", 0, 500, (0, 500), step=10, key="seg_age")

    def en_rango(serie, rango, tope, factor=1.0):
        """El extremo superior del control significa 'ese valor o más'."""
        return (serie >= rango[0] * factor) & ((serie <= rango[1] * factor) | (rango[1] >= tope))

    seg = (en_rango(train["DEBTINC"], r_debt, 60.0) & en_rango(train["VALUE"], r_val, 400, 1000)
           & en_rango(train["CLAGE"], r_age, 500)).values
    n_seg = int(seg.sum())
    if n_seg < 30:
        st.warning(f"El segmento tiene solo {n_seg} clientes: es muy chico para sacar conclusiones. Ampliá los rangos.")
    else:
        tau_seg = tau_tr[seg]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clientes en el segmento", miles(n_seg), f"{pct(n_seg / len(train))} de train", delta_color="off", delta_arrow="off")
        dif_seg = float(tau_seg.mean() - tau_tr.mean())
        c2.metric("Efecto medio en el segmento", pp(tau_seg.mean()), f"{pp(dif_seg)} vs. toda la muestra",
                  delta_color="inverse" if abs(dif_seg) >= 0.0005 else "off")
        c3.metric("Defaults evitados cada 1.000 créditos", f"{-1000 * tau_seg.mean():.0f}")
        c4.metric("Con efecto protector concluyente", pct((tau_tr_sup[seg] < 0).mean()),
                  "IC 95 % entero bajo cero", delta_color="off", delta_arrow="off")

        hist = go.Figure()
        hist.add_histogram(x=tau_seg, xbins=dict(start=-0.30, end=0.20, size=0.01), marker_color=AZUL,
                           hovertemplate="Efecto: %{x:+.0%}<br>Clientes: %{y}<extra></extra>")
        hist.add_vline(x=0, line_color=TINTA, line_width=1)
        hist.add_vline(x=float(tau_seg.mean()), line_dash="dot", line_color=NARANJA,
                       annotation_text="media del segmento", annotation_font_color=NARANJA,
                       annotation_position="top left")
        hist.update_xaxes(title="Efecto individual estimado sobre la probabilidad de default", tickformat="+.0%",
                          range=[-0.30, 0.20])
        hist.update_yaxes(title="Clientes")
        hist.update_layout(bargap=0.05)
        st.plotly_chart(estilo(hist, 300, leyenda=False), width="stretch")
        st.caption(
            f"A la izquierda del cero, la política baja el default. El promedio del bosque causal en toda la muestra "
            f"({pp(tau_tr.mean())}) es más chico que el ATE de la regresión ({pp(ate_reg)}): son dos métodos "
            "distintos y el bosque es más conservador. Coinciden en el signo, no en la magnitud, y por eso conviene "
            "leer estos números como un orden de magnitud y no como una cifra exacta.")

    with st.expander("Advertencias para leer estos resultados"):
        med = causal["mediator_audit"]
        st.markdown(
            "- **No es un experimento.** El efecto se estima con datos observacionales y vale solo si el DAG está "
            "bien especificado y no quedan confusores sin medir.\n"
            f"- **El mediador importa.** Si se ajusta por el monto del préstamo (LOAN), el efecto baja de "
            f"{pp(med['ate_total'])} a {pp(med['ate_direct_with_loan'])}, porque se bloquea parte del camino por el "
            "que actúa el tratamiento. Por eso LOAN queda fuera del ajuste.\n"
            "- **Predecir no es intervenir.** Las variables más importantes del modelo predictivo no son "
            "necesariamente las palancas que conviene mover.")
