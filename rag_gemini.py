import os
from pathlib import Path
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="google.api_core"
)


# -------------- CONFIGURACIÓN -----------------

# Use absolute path to chroma_db (project root)
CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
TEXT_COLLECTION = "nutricion_textos"
IMAGE_COLLECTION = "nutricion_imagenes"

GEMINI_MODEL = "gemini-2.0-flash"  # o "gemini-1.5-flash" según disponibilidad

# ----------------------------------------------

def init_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY en .env")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    return model

def get_chroma_collections():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    col_text = client.get_collection(TEXT_COLLECTION)
    col_images = client.get_collection(IMAGE_COLLECTION)

    return col_text, col_images

def retrieve_context(user_query, col_text, col_images, k_text=2, k_img=1):
    # búsqueda en textos
    res_text = col_text.query(
        query_texts=[user_query],
        n_results=k_text
    )

    # búsqueda en captions de imágenes
    res_img = col_images.query(
        query_texts=[user_query],
        n_results=k_img
    )

    # --- FILTRO POR UMBRAL DE SIMILITUD ---
    # tomamos distancias, documentos y metadatos
    img_docs = res_img.get("documents", [[]])[0]
    img_metas = res_img.get("metadatas", [[]])[0]
    img_dists = res_img.get("distances", [[]])[0]

    # define tu umbral (ajústalo según tu dataset)
    SIM_THRESHOLD = 0.9

    filtered = [
        (doc, meta)
        for doc, meta, dist in zip(img_docs, img_metas, img_dists)
        if dist < SIM_THRESHOLD
    ]

    # si no pasa ninguna imagen el umbral → dejar contexto vacío
    if filtered:
        img_docs, img_metas = zip(*filtered)
    else:
        img_docs, img_metas = [], []

    # Construir contexto de texto
    text_chunks = res_text.get("documents", [[]])[0]
    text_metas = res_text.get("metadatas", [[]])[0]

    ctx_text = ""
    for doc, meta in zip(text_chunks, text_metas):
        print(meta)
        src = meta.get("source_file", "desconocido")
        ctx_text += f"- [Fuente: {src}]\n{doc}\n\n"

    # Construir contexto de imágenes (captions)
    img_chunks = res_img.get("documents", [[]])[0]
    img_metas = res_img.get("metadatas", [[]])[0]

    ctx_img = ""
    for caption, meta in zip(img_chunks, img_metas):
        img_path = meta.get("source_image", "desconocida")
        ctx_img += f"- [Imagen: {img_path}]\nDescripción: {caption}\n\n"

    return ctx_text.strip(), ctx_img.strip()

def build_prompt(user_query, ctx_text, ctx_img):
    prompt = f"""
Eres un asistente de salud y bienestar especializado en nutrición y hábitos saludables.

Tu objetivo es proporcionar información clara, confiable y centrada en la salud integral de la persona. 
Comunica con calidez, empatía y de manera accesible, como lo haría un orientador de salud certificado.

=== REGLA FUNDAMENTAL ===
⚠️ SOLO puedes utilizar información que aparezca en el contexto proporcionado.
NO inventes, NO especules, NO des consejos basados en conocimiento general.
Si el contexto no contiene la información solicitada, sé honesto y transparente sobre ello.

### CONTEXTO DISPONIBLE

📖 INFORMACIÓN TEXTUAL:
{ctx_text if ctx_text else '(No hay información textual disponible para esta consulta)'}

🖼️ REFERENCIAS VISUALES:
{ctx_img if ctx_img else '(No hay referencias visuales disponibles)'}

### PREGUNTA DEL USUARIO
{user_query}

### INSTRUCCIONES PARA TU RESPUESTA

1. **Fundamentación**: Siempre basa tu respuesta exclusivamente en el contexto anterior.

2. **Estructura**: Organiza la respuesta de manera clara:
   - Saludo breve y empático
   - Respuesta directa a la pregunta
   - Citas específicas del contexto si es relevante
   - Referencias a imágenes o ejemplos si ayudan a ilustrar

3. **Tono**: Amigable, profesional y orientado a la salud
   - Utiliza un lenguaje accesible
   - Evita tecnicismos innecesarios
   - Sé motivador sin ser alarmista

4. **Limitaciones**: 
   - Si la información no está en el contexto, di claramente: "La información sobre [tema] no está disponible en mis recursos actuales."
   - NO hagas suposiciones o inferencias fuera del contexto
   - NO proporciones diagnósticos médicos generales

5. **Formato**: 
   - Responde siempre en ESPAÑOL
   - Usa bullet points o numeración cuando sea apropiado
   - Mantén la respuesta concisa pero completa

6. **Validación**: Verifica que CADA afirmación en tu respuesta esté respaldada por el contexto.
"""
    return prompt

def rag_answer(user_query):
    # Inicializar Gemini
    model = init_gemini()

    # Conectar a Chroma
    col_text, col_images = get_chroma_collections()

    # Recuperar contexto
    ctx_text, ctx_img = retrieve_context(user_query, col_text, col_images)

    # Construir prompt
    prompt = build_prompt(user_query, ctx_text, ctx_img)

    # Llamar a Gemini
    response = model.generate_content(prompt)

    return {
        "query": user_query,
        "context_text": ctx_text,
        "context_images": ctx_img,
        "answer": response.text
    }

def main():
    print("=== Demo RAG con Chroma + Gemini ===")
    print("Escribe una pregunta (en español). Ctrl+C para salir.\n")

    while True:
        try:
            user_query = input("Tú: ")
            if not user_query.strip():
                continue

            result = rag_answer(user_query)

            print("\n--- CONTEXTO TEXTO ---")
            print(result["context_text"][:500], "\n")  # mostramos solo un poco

            print("--- CONTEXTO IMÁGENES ---")
            print(result["context_images"][:500], "\n")

            print("--- RESPUESTA GEMINI ---")
            print(result["answer"])
            print("\n" + "="*60 + "\n")

        except KeyboardInterrupt:
            print("\nSaliendo...")
            break

if __name__ == "__main__":
    main()
