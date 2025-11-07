"""
Taller IA: Aplicación Multimodal con OCR y LLMs
Curso: Inteligencia Artificial
Universidad: EAFIT
Profesor: Jorge Padilla
"""

import streamlit as st
import easyocr
from PIL import Image
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv
import re

from transformers import pipeline

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
load_dotenv(dotenv_path=".env")

st.set_page_config(
    page_title="Taller IA: OCR + LLM",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Taller IA: OCR + LLM")
st.markdown("### Aplicación Multimodal con Visión Artificial y Procesamiento de Lenguaje Natural")
st.markdown("---")

# =============================================================================
# HELPERS: Hugging Face Pipelines Locales
# =============================================================================
DEFAULT_HF_MAX_TOKENS = 256

def _clean_ticks(s: str) -> str:
    """Limpia horas/timestamps tipo 10:30 y normaliza bullets/espacios."""
    s = re.sub(r"\b\d{1,2}:\d{2}\b", "", s)
    s = re.sub(r"[•\-\u2022]+\s*", "• ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

@st.cache_resource(show_spinner=False)
def _local_summarizer_fast():
    return pipeline("summarization", model="facebook/bart-base")

def hf_summarize(text, max_tokens=DEFAULT_HF_MAX_TOKENS):
    try:
        pipe = _local_summarizer_fast()
        out = pipe(text, max_length=min(256, int(max_tokens)), do_sample=False)
        return out[0]["summary_text"] if out else ""
    except Exception as e:
        st.error(f"Resumen rápido falló: {e}")
        return ""

@st.cache_resource(show_spinner=False)
def _local_ner_fast():
    return pipeline(
        "token-classification",
        model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
        aggregation_strategy="simple"
    )

def hf_entities(text, max_tokens=DEFAULT_HF_MAX_TOKENS):
    try:
        ner = _local_ner_fast()
        ents = ner(text)
        cat_map = {
            "PER": "PERSONA",
            "ORG": "ORGANIZACIÓN",
            "LOC": "LUGAR",
            "MISC": "OTRA",
            "DATE": "FECHA",
        }
        lines = [f"• [{cat_map.get(e.get('entity_group', 'MISC'))}]: {e.get('word','').strip()}"
                 for e in ents if e.get('word')]
        return "\n".join(lines) if lines else "• [INFO]: No se detectaron entidades claras."
    except Exception as e:
        st.error(f"NER rápido falló: {e}")
        return ""

@st.cache_resource(show_spinner=False)
def _local_translator_es_en_fast():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

def hf_translate_to_english(text, max_tokens=DEFAULT_HF_MAX_TOKENS):
    try:
        translator = _local_translator_es_en_fast()
        out = translator(text, max_length=min(256, int(max_tokens)))
        return out[0]["translation_text"] if out else ""
    except Exception as e:
        st.error(f"Traducción local falló: {e}")
        return ""

# =============================================================================
# MÓDULO 1: OCR
# =============================================================================
st.header("📸 Módulo 1: Extracción de Texto (OCR)")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['es', 'en'])

with st.spinner("Cargando modelo OCR..."):
    reader = load_ocr_reader()

uploaded_file = st.file_uploader(
    "Sube una imagen con texto",
    type=['png', 'jpg', 'jpeg'],
    help="Formatos soportados: PNG, JPG, JPEG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)
    image_np = np.array(image)

    if st.button("Extraer Texto", type="primary"):
        with st.spinner("Extrayendo texto de la imagen..."):
            result = reader.readtext(image_np)
            extracted_text = "\n".join([d[1] for d in result])
            st.session_state['extracted_text'] = extracted_text

    if 'extracted_text' in st.session_state:
        st.success("✅ Texto extraído exitosamente")
        st.text_area("Texto extraído:", value=st.session_state['extracted_text'], height=200)

st.markdown("---")

# =============================================================================
# MÓDULO 2 Y 3: LLMs (GROQ y HUGGING FACE)
# =============================================================================
st.header("🧩 Módulo 2 y 3: Análisis con Modelos de Lenguaje")

if 'extracted_text' not in st.session_state or not st.session_state['extracted_text']:
    st.info("👆 Primero extrae texto de una imagen en la sección superior.")
else:
    text_input = st.session_state['extracted_text']
    provider = st.radio("Proveedor:", ["GROQ", "Hugging Face"])

    temperature = st.slider("Creatividad (temperature):", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Máx. tokens (longitud):", 50, 2000, 500, 50)
    st.markdown("---")

    if provider == "GROQ":
        st.subheader("💬 Análisis con GROQ (llama-3.1-8b-instant)")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("❌ No se encontró GROQ_API_KEY en .env")
        else:
            task = st.selectbox(
                "Tarea a realizar:",
                ["Resumir texto", "Identificar entidades", "Traducir al inglés"]
            )
            if st.button("Ejecutar análisis GROQ", type="primary"):
                system_prompts = {
                    "Resumir texto": "Resume el siguiente texto en 3 puntos clave concisos:",
                    "Identificar entidades": "Extrae las entidades principales (personas, lugares, organizaciones, fechas):",
                    "Traducir al inglés": "Traduce el siguiente texto al inglés:"
                }
                client = Groq(api_key=groq_api_key)
                try:
                    with st.spinner("Analizando con GROQ..."):
                        chat = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": system_prompts[task]},
                                {"role": "user", "content": text_input}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        response = chat.choices[0].message.content
                        st.subheader("🧠 Respuesta del modelo:")
                        st.write(response)
                        st.info(f"Modelo: llama-3.1-8b-instant | Tarea: {task}")
                except Exception as e:
                    st.error(f"Error al conectar con GROQ: {e}")

    elif provider == "Hugging Face":
        st.subheader("🤗 Análisis con Hugging Face")
        hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_api_key:
            st.error("❌ No se encontró HUGGINGFACEHUB_API_TOKEN en .env")
        else:
            task = st.selectbox(
                "Tarea a realizar:",
                ["Resumir texto", "Identificar entidades", "Traducir al inglés"]
            )
            if st.button("Ejecutar análisis Hugging Face", type="primary"):
                try:
                    with st.spinner("Analizando con Hugging Face..."):
                        if task == "Resumir texto":
                            output = hf_summarize(text_input, max_tokens)
                        elif task == "Identificar entidades":
                            output = hf_entities(text_input, max_tokens)
                        elif task == "Traducir al inglés":
                            output = hf_translate_to_english(text_input, max_tokens)
                    st.subheader("🧠 Resultado del análisis:")
                    st.write(output)
                    st.info(f"Modelo utilizado: {task}")
                except Exception as e:
                    st.error(f"Error al usar Hugging Face: {e}")

# =============================================================================
# SIDEBAR: Información
# =============================================================================
with st.sidebar:
    st.header(" Información del Proyecto")
    st.markdown("""
    **Taller IA: Aplicación Multimodal con OCR y LLMs**
    
    1. Sube una imagen con texto.  
    2. Extrae el texto con OCR.  
    3. Analiza con GROQ o Hugging Face.  

    **Modelos:**
    - GROQ → `llama-3.1-8b-instant`
    - Hugging Face →  
        🧾 `facebook/bart-base` (resumen)  
        🧍 `Davlan/distilbert-base-multilingual-cased-ner-hrl` (entidades)  
        🌍 `Helsinki-NLP/opus-mt-es-en` (traducción)
    """)

    st.markdown("---")
    groq_key = os.getenv("GROQ_API_KEY")
    hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if groq_key:
        st.success("GROQ configurado")
    else:
        st.error("GROQ no configurado")

    if hf_key:
        st.success("Hugging Face configurado")
    else:
        st.error("Hugging Face no configurado")
