# 🧠 NLP & LLMs Avanzado — EAFIT 2026-1

Aplicación Streamlit para el Parcial de NLP & LLMs Avanzado.
**Docente:** Jorge Iván Padilla-Buriticá | **Universidad:** EAFIT

---

## 🗂️ Estructura del Proyecto

```
├── app.py              # Aplicación principal Streamlit
├── requirements.txt    # Dependencias
├── .streamlit/
│   └── secrets.toml    # API Key (NO subir al repo)
└── README.md
```

---

## 🚀 Instalación y Ejecución

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd <carpeta>

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar API Key (NO escribirla en código)
mkdir -p .streamlit
echo '[general]\nGROQ_API_KEY = "gsk_TU_CLAVE_AQUI"' > .streamlit/secrets.toml

# 4. Ejecutar
streamlit run app.py
```

---

## 📋 Descripción de las Partes

### ⚗️ Parte 02 — Laboratorio de Parámetros
- Panel interactivo con 6 sliders: Temperatura, Top-p, Top-k, Max Tokens, Frequency Penalty, Presence Penalty
- Experimento comparativo con 4 configuraciones contrastantes en columnas paralelas
- Gráficas Plotly de longitud en tokens y diversidad léxica (TTR)
- Campo de observaciones para documentar los resultados

### 📐 Parte 03 — Métricas de Similitud
- **Similitud Coseno** con `sentence-transformers` (all-MiniLM-L6-v2)
- **BLEU Score** con `nltk`
- **ROUGE-L** con `rouge-score`
- **BERTScore** con `bert-score`
- **LLM-as-Judge** con llamada secundaria a la API devolviendo JSON estructurado
- Radar chart con todas las métricas normalizadas

### 🤖 Parte 04 — Agente Conversacional (TutorML)
- **Personalidad:** Tutor experto en Machine Learning
- Memoria de conversación con `st.session_state`
- 5 métricas de producción en tiempo real: Latencia, TPS, Tokens entrada/salida, Costo USD, LLM-Judge
- Gráficas de línea del historial de métricas por turno
- Botón para limpiar conversación
- Controles de temperatura y max_tokens en sidebar

---

## 📸 Capturas de Pantalla
> Agregar capturas después de ejecutar la app.

---

## 🔒 Seguridad
- La API Key **nunca** se escribe en código plano
- Se usa `.streamlit/secrets.toml` o input tipo `password`
- El archivo `secrets.toml` está en `.gitignore`

---

## 📦 Librerías Principales

| Categoría | Librería |
|-----------|----------|
| API LLM | `groq` |
| UI | `streamlit` |
| Métricas NLP | `nltk`, `rouge-score`, `bert-score`, `sentence-transformers` |
| Visualización | `plotly`, `pandas` |
| Utilidades | `time`, `json`, `re` |
