"""
Parcial NLP & LLMs Avanzado – EAFIT 2026-1
Partes 02, 03 y 04: Laboratorio de Parámetros, Métricas de Similitud y Agente Conversacional
Autor: [Tu nombre]
Docente: Jorge Iván Padilla-Buriticá
"""

import streamlit as st
import time
import json
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP & LLMs Avanzado – EAFIT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0f0f1a;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        color: #a0a0c0;
        border-radius: 8px;
        padding: 8px 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        border: 1px solid #2a2a4a;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6c63ff, #48cfad) !important;
        color: white !important;
        border: none !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #6c63ff44;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-card h3 {
        font-family: 'JetBrains Mono', monospace;
        color: #48cfad;
        font-size: 1.6rem;
        margin: 0;
    }
    .metric-card p {
        color: #a0a0c0;
        font-size: 0.78rem;
        margin: 4px 0 0;
    }
    .response-box {
        background: #0f0f1a;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 14px;
        font-size: 0.82rem;
        color: #d0d0e8;
        min-height: 200px;
        font-family: 'Sora', sans-serif;
        line-height: 1.6;
    }
    .section-title {
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 1.4rem;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .section-sub {
        color: #7070a0;
        font-size: 0.82rem;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6c63ff, #48cfad);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        padding: 10px 24px;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; }
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        color: #d0d0e8;
        border-radius: 8px;
        font-family: 'Sora', sans-serif;
    }
    .stSlider { padding: 4px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar: API Key y configuración global
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 API Key")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Tu clave de Groq. Nunca se almacena en código.",
    )
    # Prioridad: secrets > input manual
    GROQ_API_KEY = (
        st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    ) or api_key_input

    st.markdown("---")
    st.markdown("**Modelo:** `llama-3.3-70b-versatile`")
    st.markdown("**API:** Groq")
    st.markdown("---")
    st.caption("Parcial NLP & LLMs Avanzado\nEAFIT 2026-1")


def get_client() -> Groq | None:
    """Retorna cliente Groq si hay API key disponible."""
    if not GROQ_API_KEY:
        st.warning("⚠️ Ingresa tu Groq API Key en el panel lateral.")
        return None
    return Groq(api_key=GROQ_API_KEY)


# ─────────────────────────────────────────────
# Pestañas principales
# ─────────────────────────────────────────────
st.markdown("## 🧠 NLP & LLMs Avanzado — EAFIT 2026‑1")
tab2, tab3, tab4 = st.tabs([
    "⚗️ Laboratorio de Parámetros",
    "📐 Métricas de Similitud",
    "🤖 Agente Conversacional",
])


# ══════════════════════════════════════════════════════════════════
# PARTE 02 – Laboratorio de Parámetros
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">⚗️ Laboratorio de Sintonización de Parámetros</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Experimenta con los hiperparámetros de generación y observa su efecto sobre las respuestas del modelo.</div>', unsafe_allow_html=True)

    # ── Panel de control ──
    with st.expander("🎛️ Panel de Control de Parámetros", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.slider("🌡️ Temperatura", 0.0, 2.0, 0.7, 0.05,
                             help="Creatividad vs. determinismo")
            top_p = st.slider("🎯 Top-p (nucleus)", 0.0, 1.0, 0.9, 0.05,
                              help="Masa de probabilidad acumulada")
        with col2:
            top_k = st.number_input("🔢 Top-k", 1, 100, 40,
                                    help="Vocabulario efectivo por paso")
            max_tokens = st.slider("📏 Max Tokens", 50, 2048, 512, 50,
                                   help="Longitud máxima de respuesta")
        with col3:
            freq_penalty = st.slider("🔁 Frequency Penalty", 0.0, 2.0, 0.0, 0.1,
                                     help="Penaliza repetición de tokens frecuentes")
            pres_penalty = st.slider("🆕 Presence Penalty", 0.0, 2.0, 0.0, 0.1,
                                     help="Penaliza aparición previa de tokens")

    prompt_libre = st.text_area(
        "✏️ Prompt libre",
        value="Explica el concepto de atención en transformers.",
        height=80,
    )

    if st.button("▶ Generar con parámetros actuales"):
        client = get_client()
        if client:
            with st.spinner("Generando respuesta..."):
                t0 = time.time()
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt_libre}],
                    temperature=temp,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=freq_penalty,
                    presence_penalty=pres_penalty,
                )
                latency = time.time() - t0
                text = resp.choices[0].message.content
                usage = resp.usage

            c1, c2, c3 = st.columns(3)
            tps = usage.completion_tokens / latency if latency > 0 else 0
            c1.metric("⏱️ Latencia", f"{latency:.2f}s")
            c2.metric("⚡ Tokens/s", f"{tps:.1f}")
            c3.metric("🔤 Tokens generados", usage.completion_tokens)
            st.markdown(f'<div class="response-box">{text}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Experimento Comparativo ──
    st.markdown("### 🔬 Experimento Comparativo — 4 Configuraciones")
    CONFIGS = [
        {"label": "T=0.1 / p=0.9", "temperature": 0.1, "top_p": 0.9},
        {"label": "T=1.5 / p=0.9", "temperature": 1.5, "top_p": 0.9},
        {"label": "T=0.1 / p=0.3", "temperature": 0.1, "top_p": 0.3},
        {"label": "T=1.5 / p=0.3", "temperature": 1.5, "top_p": 0.3},
    ]
    FIXED_PROMPT = "Explica el concepto de atención en transformers."
    st.caption(f"**Prompt fijo:** _{FIXED_PROMPT}_")

    if st.button("🚀 Ejecutar experimento comparativo"):
        client = get_client()
        if client:
            results = []
            cols = st.columns(4)
            prog = st.progress(0, "Generando respuestas...")

            for i, cfg in enumerate(CONFIGS):
                prog.progress((i + 1) / 4, f"Configuración {i+1}/4: {cfg['label']}")
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": FIXED_PROMPT}],
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                    max_tokens=400,
                )
                text = resp.choices[0].message.content
                tokens = len(text.split())
                unique = len(set(text.lower().split()))
                ttr = unique / tokens if tokens > 0 else 0
                results.append({
                    "config": cfg["label"],
                    "text": text,
                    "tokens": tokens,
                    "ttr": round(ttr, 3),
                })
                with cols[i]:
                    st.markdown(f"**{cfg['label']}**")
                    st.markdown(f'<div class="response-box" style="font-size:0.75rem">{text[:600]}{"..." if len(text)>600 else ""}</div>', unsafe_allow_html=True)

            prog.empty()
            st.session_state["exp_results"] = results

    # ── Gráficas post-experimento ──
    if "exp_results" in st.session_state:
        results = st.session_state["exp_results"]
        df = pd.DataFrame(results)

        gc1, gc2 = st.columns(2)
        with gc1:
            fig_tok = px.bar(
                df, x="config", y="tokens",
                title="📊 Longitud en Tokens por Configuración",
                color="config",
                color_discrete_sequence=["#6c63ff", "#48cfad", "#f7971e", "#f64f59"],
                template="plotly_dark",
            )
            fig_tok.update_layout(showlegend=False, paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a")
            st.plotly_chart(fig_tok, use_container_width=True)

        with gc2:
            fig_ttr = px.bar(
                df, x="config", y="ttr",
                title="🔤 Diversidad Léxica (Type-Token Ratio)",
                color="config",
                color_discrete_sequence=["#6c63ff", "#48cfad", "#f7971e", "#f64f59"],
                template="plotly_dark",
            )
            fig_ttr.update_layout(showlegend=False, paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a")
            st.plotly_chart(fig_ttr, use_container_width=True)

        st.markdown("### 📝 Documenta tus observaciones")
        st.text_area(
            "Análisis del experimento comparativo",
            placeholder="Ej: Con temperatura alta y top-p amplio (T=1.5/p=0.9) se observa mayor diversidad léxica pero menor coherencia...",
            height=120,
            key="obs_exp",
        )


# ══════════════════════════════════════════════════════════════════
# PARTE 03 – Métricas de Similitud
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📐 Métricas de Similitud y Evaluación Automática</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Compara cuantitativamente un texto de referencia contra la salida generada por el LLM.</div>', unsafe_allow_html=True)

    col_ref, col_gen = st.columns(2)
    with col_ref:
        ref_text = st.text_area(
            "📄 Texto de referencia (ground truth)",
            height=160,
            placeholder="Escribe aquí el texto de referencia...",
        )
    with col_gen:
        gen_prompt = st.text_area(
            "✏️ Prompt para generar respuesta candidata",
            height=160,
            placeholder="Escribe el prompt que se enviará al LLM...",
        )

    if st.button("📊 Calcular métricas"):
        client = get_client()
        if not ref_text.strip():
            st.error("Ingresa un texto de referencia.")
        elif not gen_prompt.strip():
            st.error("Ingresa un prompt.")
        elif client:
            # ── Generar texto candidato ──
            with st.spinner("Generando respuesta del LLM..."):
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": gen_prompt}],
                    temperature=0.5,
                    max_tokens=512,
                )
                gen_text = resp.choices[0].message.content

            st.markdown("#### 💬 Respuesta generada")
            st.markdown(f'<div class="response-box">{gen_text}</div>', unsafe_allow_html=True)
            st.markdown("---")

            scores = {}

            # ── Similitud Coseno con sentence-transformers ──
            try:
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                @st.cache_resource
                def load_st_model():
                    """Carga el modelo sentence-transformers (cacheado)."""
                    return SentenceTransformer("all-MiniLM-L6-v2")

                with st.spinner("Calculando similitud coseno..."):
                    model_st = load_st_model()
                    emb_ref = model_st.encode([ref_text])
                    emb_gen = model_st.encode([gen_text])
                    cosine_sim = float(cosine_similarity(emb_ref, emb_gen)[0][0])
                    scores["Coseno"] = round(cosine_sim, 4)
            except ImportError:
                st.warning("Instala `sentence-transformers` y `scikit-learn` para similitud coseno.")
                scores["Coseno"] = 0.0

            # ── BLEU Score ──
            try:
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

                def compute_bleu(reference: str, candidate: str) -> float:
                    """Calcula BLEU score entre referencia y candidato."""
                    ref_tokens = [reference.lower().split()]
                    cand_tokens = candidate.lower().split()
                    smoother = SmoothingFunction().method1
                    return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoother)

                scores["BLEU"] = round(compute_bleu(ref_text, gen_text), 4)
            except ImportError:
                st.warning("Instala `nltk` para BLEU.")
                scores["BLEU"] = 0.0

            # ── ROUGE-L ──
            try:
                from rouge_score import rouge_scorer

                def compute_rouge_l(reference: str, candidate: str) -> float:
                    """Calcula ROUGE-L (LCS) entre referencia y candidato."""
                    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                    result = scorer.score(reference, candidate)
                    return result["rougeL"].fmeasure

                scores["ROUGE-L"] = round(compute_rouge_l(ref_text, gen_text), 4)
            except ImportError:
                st.warning("Instala `rouge-score` para ROUGE-L.")
                scores["ROUGE-L"] = 0.0

            # ── BERTScore ──
            try:
                from bert_score import score as bert_score_fn

                @st.cache_data(show_spinner=False)
                def compute_bert_score(ref: str, cand: str) -> float:
                    """Calcula BERTScore F1 entre referencia y candidato."""
                    P, R, F1 = bert_score_fn([cand], [ref], lang="es", verbose=False)
                    return float(F1.mean())

                with st.spinner("Calculando BERTScore (puede tardar un momento)..."):
                    scores["BERTScore"] = round(compute_bert_score(ref_text, gen_text), 4)
            except ImportError:
                st.warning("Instala `bert-score` para BERTScore.")
                scores["BERTScore"] = 0.0

            # ── LLM-as-Judge ──
            judge_system = """Eres un evaluador experto en NLP. Evalúa la respuesta generada
comparándola con la referencia. Responde ÚNICAMENTE en JSON con este esquema:
{
  "score": <número 1-10>,
  "veracidad": <número 1-10>,
  "coherencia": <número 1-10>,
  "relevancia": <número 1-10>,
  "fortalezas": "<texto>",
  "debilidades": "<texto>"
}"""
            judge_user = f"""REFERENCIA: {ref_text}
RESPUESTA GENERADA: {gen_text}
PROMPT ORIGINAL: {gen_prompt}"""

            with st.spinner("Ejecutando LLM-as-Judge..."):
                judge_resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": judge_system},
                        {"role": "user", "content": judge_user},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )
                judge_raw = judge_resp.choices[0].message.content

            # Parsear JSON del juez
            judge_data = {}
            try:
                clean = re.sub(r"```json|```", "", judge_raw).strip()
                judge_data = json.loads(clean)
                scores["LLM-Judge"] = judge_data.get("score", 5) / 10
            except Exception:
                scores["LLM-Judge"] = 0.5
                st.warning("No se pudo parsear el JSON del juez.")

            # ── Mostrar métricas ──
            st.markdown("### 📏 Resultados de Métricas")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            metrics_display = [
                (mc1, "🔵 Coseno", scores.get("Coseno", 0)),
                (mc2, "📘 BLEU", scores.get("BLEU", 0)),
                (mc3, "📗 ROUGE-L", scores.get("ROUGE-L", 0)),
                (mc4, "🤖 BERTScore", scores.get("BERTScore", 0)),
                (mc5, "⚖️ LLM-Judge", scores.get("LLM-Judge", 0)),
            ]
            for col, label, val in metrics_display:
                col.metric(label, f"{val:.4f}")

            # ── Radar Chart ──
            radar_labels = ["Coseno", "BLEU", "ROUGE-L", "BERTScore", "LLM-Judge"]
            radar_vals = [scores.get(k, 0) for k in radar_labels]
            radar_vals_closed = radar_vals + [radar_vals[0]]
            theta_closed = radar_labels + [radar_labels[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_vals_closed,
                theta=theta_closed,
                fill="toself",
                fillcolor="rgba(108,99,255,0.25)",
                line=dict(color="#6c63ff", width=2),
                name="Métricas",
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], color="#a0a0c0"),
                    angularaxis=dict(color="#a0a0c0"),
                    bgcolor="#1a1a2e",
                ),
                paper_bgcolor="#0f0f1a",
                font=dict(color="#d0d0e8"),
                title="🕸️ Radar de Métricas Normalizadas",
                showlegend=False,
                height=420,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # ── Detalle del Juez ──
            if judge_data:
                st.markdown("### ⚖️ Evaluación LLM-as-Judge")
                jc1, jc2, jc3 = st.columns(3)
                jc1.metric("Veracidad", f"{judge_data.get('veracidad', '-')}/10")
                jc2.metric("Coherencia", f"{judge_data.get('coherencia', '-')}/10")
                jc3.metric("Relevancia", f"{judge_data.get('relevancia', '-')}/10")
                st.success(f"✅ **Fortalezas:** {judge_data.get('fortalezas', 'N/A')}")
                st.error(f"⚠️ **Debilidades:** {judge_data.get('debilidades', 'N/A')}")


# ══════════════════════════════════════════════════════════════════
# PARTE 04 – Agente Conversacional
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🤖 Agente Conversacional — Tutor de Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Agente especializado con memoria de conversación y métricas de producción en tiempo real.</div>', unsafe_allow_html=True)

    # System prompt del agente
    AGENT_SYSTEM = """Eres TutorML, un agente experto en Machine Learning y Deep Learning.
Tu personalidad: didáctico, preciso, usa ejemplos concretos y analogías claras.
Tu dominio: regresión, clasificación, redes neuronales, NLP, LLMs, métricas de evaluación.
Restricciones: Solo respondes preguntas relacionadas con ML/IA. Si te preguntan algo fuera de tu
dominio, redirige amablemente al tema de ML. Responde siempre en español, de forma estructurada
y con ejemplos de código Python cuando sea útil."""

    # Inicializar estado de sesión
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []
    if "agent_metrics" not in st.session_state:
        st.session_state.agent_metrics = []

    # ── Controles laterales del agente ──
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 Configuración del Agente")
        agent_temp = st.slider("🌡️ Temperatura", 0.0, 2.0, 0.7, 0.05, key="agent_temp")
        agent_max_tokens = st.slider("📏 Max Tokens", 100, 2048, 800, 50, key="agent_maxtok")
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.agent_history = []
            st.session_state.agent_metrics = []
            st.rerun()

    # ── Render del historial ──
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.agent_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Input del usuario ──
    user_input = st.chat_input("Pregúntale a TutorML sobre Machine Learning...")

    if user_input:
        client = get_client()
        if client:
            # Mostrar mensaje del usuario
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.agent_history.append({"role": "user", "content": user_input})

            # Construir mensajes con historial
            messages = [{"role": "system", "content": AGENT_SYSTEM}] + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.agent_history
            ]

            # Generar respuesta con medición de latencia
            t0 = time.time()
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=agent_temp,
                max_tokens=agent_max_tokens,
            )
            latency = time.time() - t0
            assistant_text = resp.choices[0].message.content
            usage = resp.usage

            # Mostrar respuesta
            with st.chat_message("assistant"):
                st.markdown(assistant_text)
            st.session_state.agent_history.append({"role": "assistant", "content": assistant_text})

            # ── Calcular métricas ──
            tps = usage.completion_tokens / latency if latency > 0 else 0

            # Costo estimado Groq llama-3.3-70b: ~$0.59 / 1M input, ~$0.79 / 1M output
            cost = (usage.prompt_tokens * 0.59 + usage.completion_tokens * 0.79) / 1_000_000

            # LLM-Judge automático
            judge_sys = """Evalúa la respuesta del tutor. Responde SOLO con un JSON:
{"score": <1-10>, "veracidad": <1-10>, "coherencia": <1-10>, "relevancia": <1-10>}"""
            judge_u = f"PREGUNTA: {user_input}\nRESPUESTA: {assistant_text}"
            try:
                jr = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": judge_sys},
                        {"role": "user", "content": judge_u},
                    ],
                    temperature=0.1,
                    max_tokens=100,
                )
                jraw = re.sub(r"```json|```", "", jr.choices[0].message.content).strip()
                jdata = json.loads(jraw)
                judge_score = jdata.get("score", 7)
            except Exception:
                judge_score = 7

            # Guardar métricas del turno
            turn_num = len(st.session_state.agent_metrics) + 1
            st.session_state.agent_metrics.append({
                "turno": turn_num,
                "latencia_s": round(latency, 2),
                "tps": round(tps, 1),
                "tokens_entrada": usage.prompt_tokens,
                "tokens_salida": usage.completion_tokens,
                "costo_usd": round(cost, 6),
                "llm_judge": judge_score,
            })

            # ── Mostrar métricas del turno actual ──
            st.markdown("---")
            st.markdown("#### 📈 Métricas de Producción — Último Turno")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("⏱️ Latencia", f"{latency:.2f}s")
            m2.metric("⚡ Tokens/s", f"{tps:.1f}")
            m3.metric("📥 Tokens entrada", usage.prompt_tokens)
            m4.metric("📤 Tokens salida", usage.completion_tokens)
            m5.metric("💰 Costo USD", f"${cost:.6f}")
            st.metric("⚖️ LLM-Judge Score", f"{judge_score}/10")

    # ── Historial de métricas como gráfica de línea ──
    if st.session_state.agent_metrics:
        st.markdown("---")
        st.markdown("### 📊 Historial de Métricas por Turno")
        df_metrics = pd.DataFrame(st.session_state.agent_metrics)

        tab_m1, tab_m2, tab_m3 = st.tabs(["⏱️ Latencia & TPS", "🔤 Tokens", "⚖️ LLM-Judge"])

        with tab_m1:
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Scatter(
                x=df_metrics["turno"], y=df_metrics["latencia_s"],
                mode="lines+markers", name="Latencia (s)",
                line=dict(color="#6c63ff", width=2),
                marker=dict(size=8),
            ))
            fig_lat.add_trace(go.Scatter(
                x=df_metrics["turno"], y=df_metrics["tps"],
                mode="lines+markers", name="Tokens/s",
                line=dict(color="#48cfad", width=2),
                marker=dict(size=8),
                yaxis="y2",
            ))
            fig_lat.update_layout(
                title="Latencia y Tokens/segundo por Turno",
                xaxis_title="Turno",
                yaxis=dict(title="Latencia (s)", color="#6c63ff"),
                yaxis2=dict(title="Tokens/s", overlaying="y", side="right", color="#48cfad"),
                template="plotly_dark",
                paper_bgcolor="#0f0f1a",
                plot_bgcolor="#0f0f1a",
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        with tab_m2:
            fig_tok = go.Figure()
            fig_tok.add_trace(go.Bar(
                x=df_metrics["turno"], y=df_metrics["tokens_entrada"],
                name="Tokens entrada", marker_color="#6c63ff",
            ))
            fig_tok.add_trace(go.Bar(
                x=df_metrics["turno"], y=df_metrics["tokens_salida"],
                name="Tokens salida", marker_color="#48cfad",
            ))
            fig_tok.update_layout(
                barmode="group", title="Tokens de Entrada vs Salida por Turno",
                xaxis_title="Turno", yaxis_title="Tokens",
                template="plotly_dark",
                paper_bgcolor="#0f0f1a",
                plot_bgcolor="#0f0f1a",
            )
            st.plotly_chart(fig_tok, use_container_width=True)

        with tab_m3:
            fig_judge = go.Figure()
            fig_judge.add_trace(go.Scatter(
                x=df_metrics["turno"], y=df_metrics["llm_judge"],
                mode="lines+markers+text",
                text=df_metrics["llm_judge"],
                textposition="top center",
                line=dict(color="#f7971e", width=2),
                marker=dict(size=10, color="#f7971e"),
                fill="tozeroy",
                fillcolor="rgba(247,151,30,0.15)",
                name="LLM-Judge Score",
            ))
            fig_judge.update_layout(
                title="Puntuación LLM-Judge por Turno",
                xaxis_title="Turno",
                yaxis=dict(title="Score (1-10)", range=[0, 10]),
                template="plotly_dark",
                paper_bgcolor="#0f0f1a",
                plot_bgcolor="#0f0f1a",
            )
            st.plotly_chart(fig_judge, use_container_width=True)

        # Tabla resumen
        st.dataframe(
            df_metrics.rename(columns={
                "turno": "Turno",
                "latencia_s": "Latencia (s)",
                "tps": "Tokens/s",
                "tokens_entrada": "Tokens Entrada",
                "tokens_salida": "Tokens Salida",
                "costo_usd": "Costo (USD)",
                "llm_judge": "LLM-Judge",
            }),
            use_container_width=True,
            hide_index=True,
        )
