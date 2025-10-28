import streamlit as st
import easyocr
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os
from groq import Groq
from huggingface_hub import InferenceClient

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(page_title="Taller IA: OCR + LLM", layout="wide")
st.title("🤖 Taller IA: OCR + LLM")
st.markdown("---")

# ============================================
# MÓDULO 1: OCR - LECTOR DE IMÁGENES
# ============================================

st.header("📷 Módulo 1: Extracción de Texto (OCR)")

# Función para cargar el modelo OCR con caché
@st.cache_resource
def cargar_modelo_ocr():
    """Carga el modelo OCR una sola vez y lo mantiene en memoria"""
    reader = easyocr.Reader(['es', 'en'], gpu=False)
    return reader

# Cargar el modelo OCR
with st.spinner("Cargando modelo OCR..."):
    reader = cargar_modelo_ocr()

# Widget para subir imagen
archivo_imagen = st.file_uploader(
    "Sube una imagen con texto",
    type=['png', 'jpg', 'jpeg'],
    help="Formatos soportados: PNG, JPG, JPEG"
)

# Procesar imagen si se ha subido
if archivo_imagen is not None:
    # Mostrar imagen
    imagen = Image.open(archivo_imagen)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)
    
    # Convertir imagen a numpy array para OCR
    imagen_array = np.array(imagen)
    
    # Ejecutar OCR
    with st.spinner("Extrayendo texto de la imagen..."):
        resultado_ocr = reader.readtext(imagen_array)
    
    # Extraer solo el texto
    texto_extraido = " ".join([deteccion[1] for deteccion in resultado_ocr])
    
    # Guardar en session_state para persistencia
    st.session_state['texto_extraido'] = texto_extraido
    
    # Mostrar texto extraído
    st.subheader("📝 Texto Extraído:")
    st.text_area(
        "Texto detectado en la imagen",
        value=texto_extraido,
        height=150,
        key="texto_ocr"
    )
    
    st.success(f"✅ Se extrajeron {len(resultado_ocr)} fragmentos de texto")

st.markdown("---")

# ============================================
# MÓDULO 2 y 3: ANÁLISIS CON LLM
# ============================================

st.header("🧠 Módulo 2 y 3: Análisis con Modelos de Lenguaje")

# Verificar si hay texto extraído
if 'texto_extraido' in st.session_state and st.session_state['texto_extraido']:
    
    # Crear columnas para mejor organización
    col1, col2 = st.columns(2)
    
    with col1:
        # Selector de proveedor de API
        proveedor = st.radio(
            "Selecciona el proveedor de LLM:",
            ["GROQ", "Hugging Face"],
            horizontal=True
        )
    
    with col2:
        # Selector de tarea
        tarea = st.selectbox(
            "Selecciona la tarea a realizar:",
            [
                "Resumir en 3 puntos clave",
                "Identificar las entidades principales",
                "Traducir al inglés",
                "Analizar sentimiento",
                "Extraer palabras clave"
            ]
        )
    
    # Parámetros ajustables
    st.subheader("⚙️ Parámetros del Modelo")
    col3, col4 = st.columns(2)
    
    with col3:
        temperature = st.slider(
            "Temperature (Creatividad)",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Valores bajos (0.1-0.5): respuestas más precisas y determinísticas. Valores altos (0.8-2.0): respuestas más creativas y variadas."
        )
    
    with col4:
        max_tokens = st.slider(
            "Max Tokens (Longitud de respuesta)",
            min_value=50,
            max_value=1000,
            value=300,
            step=50,
            help="Controla la longitud máxima de la respuesta generada."
        )
    
    # Opciones específicas para GROQ
    if proveedor == "GROQ":
        modelo_groq = st.selectbox(
            "Selecciona el modelo de GROQ:",
            [
                "llama3-8b-8192",
                "mixtral-8x7b-32768",
                "gemma-7b-it"
            ]
        )
    
    # Botón para analizar
    if st.button("🚀 Analizar Texto", type="primary"):
        texto_analizar = st.session_state['texto_extraido']
        
        # Construir el prompt según la tarea
        prompts_tareas = {
            "Resumir en 3 puntos clave": f"Resume el siguiente texto en exactamente 3 puntos clave. Sé conciso y directo:\n\n{texto_analizar}",
            "Identificar las entidades principales": f"Identifica y lista todas las entidades principales (personas, lugares, organizaciones, fechas) en el siguiente texto:\n\n{texto_analizar}",
            "Traducir al inglés": f"Traduce el siguiente texto al inglés de manera precisa y natural:\n\n{texto_analizar}",
            "Analizar sentimiento": f"Analiza el sentimiento del siguiente texto (positivo, negativo o neutral) y explica por qué:\n\n{texto_analizar}",
            "Extraer palabras clave": f"Extrae las 5-10 palabras clave más importantes del siguiente texto:\n\n{texto_analizar}"
        }
        
        prompt_usuario = prompts_tareas[tarea]
        
        # ============================================
        # OPCIÓN 1: GROQ API
        # ============================================
        if proveedor == "GROQ":
            try:
                with st.spinner(f"Analizando con GROQ ({modelo_groq})..."):
                    # Obtener clave API
                    groq_api_key = os.getenv("GROQ_API_KEY")
                    
                    if not groq_api_key:
                        st.error("❌ No se encontró la clave API de GROQ en el archivo .env")
                    else:
                        # Instanciar cliente de GROQ
                        cliente_groq = Groq(api_key=groq_api_key)
                        
                        # Realizar llamada a la API
                        respuesta = cliente_groq.chat.completions.create(
                            model=modelo_groq,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Eres un asistente experto en análisis de texto. Responde de manera clara, concisa y profesional."
                                },
                                {
                                    "role": "user",
                                    "content": prompt_usuario
                                }
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        # Extraer y mostrar respuesta
                        resultado = respuesta.choices[0].message.content
                        
                        st.subheader("📊 Resultado del Análisis (GROQ)")
                        st.markdown(resultado)
                        
                        # Información adicional
                        st.info(f"🔹 Modelo: {modelo_groq} | Temperature: {temperature} | Max Tokens: {max_tokens}")
                        
            except Exception as e:
                st.error(f"❌ Error al conectar con GROQ: {str(e)}")
        
        # ============================================
        # OPCIÓN 2: HUGGING FACE API
        # ============================================
        elif proveedor == "Hugging Face":
            try:
                with st.spinner("Analizando con Hugging Face..."):
                    # Obtener clave API
                    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
                    
                    if not hf_api_key:
                        st.error("❌ No se encontró la clave API de Hugging Face en el archivo .env")
                    else:
                        # Instanciar cliente de Hugging Face
                        cliente_hf = InferenceClient(token=hf_api_key)
                        
                        # Seleccionar modelo según la tarea
                        if tarea == "Resumir en 3 puntos clave":
                            resultado = cliente_hf.summarization(
                                texto_analizar,
                                max_length=max_tokens,
                                min_length=50
                            )
                            resultado_texto = resultado.summary_text
                        else:
                            # Para otras tareas, usar chat completion
                            respuesta = cliente_hf.chat_completion(
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "Eres un asistente experto en análisis de texto."
                                    },
                                    {
                                        "role": "user",
                                        "content": prompt_usuario
                                    }
                                ],
                                max_tokens=max_tokens,
                                temperature=temperature
                            )
                            resultado_texto = respuesta.choices[0].message.content
                        
                        # Mostrar resultado
                        st.subheader("📊 Resultado del Análisis (Hugging Face)")
                        st.markdown(resultado_texto)
                        
                        # Información adicional
                        st.info(f"🔹 Proveedor: Hugging Face | Temperature: {temperature} | Max Tokens: {max_tokens}")
                        
            except Exception as e:
                st.error(f"❌ Error al conectar con Hugging Face: {str(e)}")
                st.info("💡 Tip: Algunas tareas pueden requerir modelos específicos o tener límites de uso en la API gratuita.")

else:
    st.info("👆 Por favor, sube una imagen primero para extraer texto.")

# ============================================
# SECCIÓN DE INFORMACIÓN Y REFLEXIÓN
# ============================================

st.markdown("---")
st.header("💭 Puntos de Reflexión")

with st.expander("🤔 Preguntas para discusión"):
    st.markdown("""
    ### Diferencias de Velocidad
    - **GROQ**: Optimizado para velocidad extrema, ideal para aplicaciones en tiempo real
    - **Hugging Face**: Mayor variedad de modelos, pero puede ser más lento dependiendo del modelo
    
    ### Efecto de Temperature
    - **Valores bajos (0.1-0.5)**: Respuestas más consistentes y determinísticas. Ideal para tareas técnicas.
    - **Valores altos (0.8-2.0)**: Respuestas más creativas y variadas. Útil para contenido creativo.
    
    ### Calidad del OCR
    - La calidad del texto extraído afecta directamente la precisión del análisis
    - Imágenes claras y con buen contraste producen mejores resultados
    - El idioma y la fuente tipográfica pueden influir en la precisión
    
    ### Extensiones Posibles
    - Análisis de sentimientos en redes sociales
    - Clasificación de documentos
    - Generación automática de Q&A
    - Extracción de datos estructurados
    - Traducción multiidioma
    """)

with st.expander("📚 Recursos y Documentación"):
    st.markdown("""
    - [Documentación de Streamlit](https://docs.streamlit.io)
    - [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
    - [GROQ Documentation](https://console.groq.com/docs)
    - [Hugging Face Inference API](https://huggingface.co/docs/api-inference)
    """)