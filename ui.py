import streamlit as st
from pathlib import Path
from urllib.parse import quote

RUTA_CSS = Path(__file__).resolve().parent / "styles" / "theme.css"

AUTORES = [
    "Jhon Andrés Díaz Cano",
    "Alejandro Castaño Uzquiano",
]

COLOR_ICONO = "#4f8f76"

ICONOS = {
    "chart": "M3 3v18h18v-2H5V3H3zm4 14h2V9H7v8zm4 0h2V5h-2v12zm4 0h2v-6h-2v6zm4 0h2v-9h-2v9z",
    "user": "M12 12a4 4 0 1 0-4-4 4 4 0 0 0 4 4zm0 2c-3.33 0-8 1.67-8 5v1h16v-1c0-3.33-4.67-5-8-5z",
    "chip": "M9 3h6v2h3a2 2 0 0 1 2 2v3h2v2h-2v2h2v2h-2v3a2 2 0 0 1-2 2h-3v2H9v-2H6a2 2 0 0 1-2-2v-3H2v-2h2v-2H2V10h2V7a2 2 0 0 1 2-2h3V3zm0 4v10h6V7H9z",
    "sliders": "M3 5h10v2H3V5zm14 0h4v2h-4V5zM3 11h4v2H3v-2zm8 0h10v2H11v-2zM3 17h12v2H3v-2zm16 0h2v2h-2v-2zM14 4h2v4h-2V4zm-8 6h2v4H6v-4zm10 6h2v4h-2v-4z",
    "check": "M9.2 16.6 4.8 12.2l1.4-1.4 3 3 8.6-8.6 1.4 1.4-10 10z",
    "clipboard": "M9 3h6a2 2 0 0 1 2 2h1a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h1a2 2 0 0 1 2-2zm0 2v2h6V5H9zm1 8 1.4 1.4L16 10.8l-1.4-1.4-3.2 3.2-.8-.8L9.2 13.2 10 14z",
    "trophy": "M7 4h10v2h3v3a5 5 0 0 1-4.1 4.9A5 5 0 0 1 13 16.9V18h3v2H8v-2h3v-1.1a5 5 0 0 1-2.9-2.99A5 5 0 0 1 4 9V6h3V4zm10 4h1v1a3 3 0 0 1-2 2.82V8zm-12 0h1v3.82A3 3 0 0 1 4 9V8z",
    "ruler": "M3 7h18v10H3V7zm2 2v2h2V9H5zm4 0v3h2V9H9zm4 0v2h2V9h-2zm4 0v3h2V9h-2z",
    "chart-bar": "M4 19h16v2H4v-2zm2-2V9h3v8H6zm5 0V5h3v12h-3zm5 0v-5h3v5h-3z",
    "car": "M5 11 6.5 6.5A2 2 0 0 1 8.4 5h7.2a2 2 0 0 1 1.9 1.5L19 11h1a1 1 0 0 1 1 1v5h-2a2.5 2.5 0 0 1-5 0H10a2.5 2.5 0 0 1-5 0H3v-5a1 1 0 0 1 1-1h1zm2.1-4 .8 2.5h8.2L16.9 7H7.1zM6.5 16.5A1 1 0 1 0 6.5 18a1 1 0 0 0 0-1.5zm11 0a1 1 0 1 0 0 1.5 1 1 0 0 0 0-1.5z",
    "table": "M3 5h18v14H3V5zm2 2v3h14V7H5zm0 5v5h6v-5H5zm8 0v5h6v-5h-6z",
    "file": "M6 2h8l4 4v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zm8 1.5V7h3.5L14 3.5zM8 11h8v2H8v-2zm0 4h8v2H8v-2z",
    "eye": "M12 5c5 0 9.3 3.1 11 7-1.7 3.9-6 7-11 7S2.7 15.9 1 12c1.7-3.9 6-7 11-7zm0 2a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm0 2.5A2.5 2.5 0 1 1 9.5 12 2.5 2.5 0 0 1 12 9.5z",
    "tags": "M10.6 3H4a1 1 0 0 0-1 1v6.6a1 1 0 0 0 .3.7l9.4 9.4a1 1 0 0 0 1.4 0l6.6-6.6a1 1 0 0 0 0-1.4L11.3 3.3a1 1 0 0 0-.7-.3zM7 8.5A1.5 1.5 0 1 1 8.5 7 1.5 1.5 0 0 1 7 8.5z",
}


def _data_uri(path_d):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        f'<path fill="{COLOR_ICONO}" d="{path_d}"/></svg>'
    )
    return "data:image/svg+xml," + quote(svg)


def _css_iconos():
    reglas = [
        """.icon {
            display: inline-block;
            width: 1.05rem;
            height: 1.05rem;
            margin-right: 0.35rem;
            vertical-align: -0.2em;
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
            flex-shrink: 0;
        }
        .section-title > .icon {
            width: 2.35rem;
            height: 2.35rem;
            margin-right: 0;
            vertical-align: middle;
            border-radius: 14px;
            background-color: var(--bg);
            background-size: 1.15rem 1.15rem;
            box-shadow: var(--neu-out-sm);
        }
        .metric-card .label .icon,
        .hero-kicker .icon,
        .author-chip .icon,
        .status-banner .icon,
        .result-banner .icon {
            width: 1rem;
            height: 1rem;
        }"""
    ]
    for nombre, path_d in ICONOS.items():
        reglas.append(f'.icon-{nombre} {{ background-image: url("{_data_uri(path_d)}"); }}')
    return "\n".join(reglas)


def _icon(nombre):
    return f'<span class="icon icon-{nombre}" aria-hidden="true">&#8203;</span>'


def aplicar_estilos():
    css = RUTA_CSS.read_text(encoding="utf-8") + "\n" + _css_iconos()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def titulo_seccion(icono, texto):
    st.markdown(
        f"""
        <div class="section-title">
            {_icon(icono)}
            {texto}
        </div>
        """,
        unsafe_allow_html=True,
    )


def texto_ayuda(html):
    st.markdown(f'<p class="hint">{html}</p>', unsafe_allow_html=True)


def panel(html):
    st.markdown(f'<div class="panel">{html}</div>', unsafe_allow_html=True)


def render_hero():
    autores_html = "".join(
        f'<span class="author-chip">{_icon("user")}{nombre}</span>'
        for nombre in AUTORES
    )
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">
                {_icon("chart")}
                Machine Learning · CRISP-DM
            </div>
            <h1>Predicción del precio de vehículos usados</h1>
            <p>
                Modelo de regresión supervisada para estimar el precio de venta
                a partir de características técnicas y comerciales.
                Incluye ajuste de hiperparámetros, evaluación y despliegue de predicción.
            </p>
            <div class="authors">{autores_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(nombre_modelo):
    st.markdown(
        f"""
        <div class="status-banner">
            {_icon("chip")}
            <div>
                <strong>Modelo en uso:</strong>
                <span>{nombre_modelo}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metricas(nombre_modelo, mae, r2):
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card accent">
                <div class="label">{_icon("trophy")} Mejor modelo</div>
                <div class="value">{nombre_modelo}</div>
            </div>
            <div class="metric-card">
                <div class="label">{_icon("ruler")} MAE</div>
                <div class="value">{mae:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">{_icon("chart-bar")} R²</div>
                <div class="value">{r2:.4f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_resultado(mensaje_html, icono="check"):
    st.markdown(
        f"""
        <div class="result-banner">
            {_icon(icono)}
            <div>{mensaje_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_precio(prediccion):
    st.markdown(
        f"""
        <div class="result-banner">
            {_icon("tags")}
            <div>
                Precio estimado
                <div class="price">${prediccion:,.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_parametros(mejores_params):
    filas = []
    for nombre, params in mejores_params.items():
        filas.append(
            f'<div class="param-row"><strong>{nombre}</strong><code>{params}</code></div>'
        )
    st.markdown("".join(filas), unsafe_allow_html=True)
