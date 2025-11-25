import os
import chromadb
from dotenv import load_dotenv

from datasets import Dataset

# Google Generative Language API
from google.ai.generativelanguage_v1beta import GenerativeServiceClient
from google.ai.generativelanguage_v1beta.types import GenerateContentRequest
from google.api_core.client_options import ClientOptions

# RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Custom wrappers
from gemini_llm import GeminiLLM
from hf_embedder import HFEmbedder


# ---------------- CONFIG ------------------

CHROMA_PATH = "./chroma_db"
TEXT_COLLECTION = "nutricion_textos"
IMAGE_COLLECTION = "nutricion_imagenes"

# Modelo de evaluación (ligero, estable, soportado)
GEMINI_MODEL = "models/gemini-1.5-flash"

# --------------------------------------------------


def init_gemini_raw():
    """Inicializa Google Gemini usando REST y API key."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY en .env")

    client = GenerativeServiceClient(
        client_options=ClientOptions(api_key=api_key)
    )

    return client


def generate_message(client, prompt):
    """Genera texto usando el cliente REST."""
    req = GenerateContentRequest(
        model=GEMINI_MODEL,
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    )

    resp = client.generate_content(request=req)
    return resp.candidates[0].content.parts[0].text


def get_collections():
    """Obtiene colecciones de Chroma."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return (
        client.get_collection(TEXT_COLLECTION),
        client.get_collection(IMAGE_COLLECTION)
    )


def retrieve_context(query, col_text, col_img, k_text=4, k_img=2, sim_threshold=1.3):
    """Recupera contexto desde Chroma (texto + captions filtrados)."""

    # TEXTOS
    t = col_text.query(query_texts=[query], n_results=k_text)
    text_docs = t["documents"][0]

    # IMÁGENES
    i = col_img.query(query_texts=[query], n_results=k_img)
    img_docs = i["documents"][0]
    img_dists = i["distances"][0]

    # Filtrar por similitud
    img_filtered = [
        doc for doc, dist in zip(img_docs, img_dists) if dist < sim_threshold
    ]

    return text_docs + img_filtered


def rag_answer(client, question, context):
    """Respuesta del RAG completa."""
    ctx = "\n".join(context)

    prompt = f"""
Responde usando únicamente el CONTEXTO dado.
Si el contexto no contiene la información, responde: "No hay suficiente información".

### CONTEXTO:
{ctx}

### PREGUNTA:
{question}

### RESPUESTA EN ESPAÑOL:
"""
    return generate_message(client, prompt)


def generate_reference_answer(client, question, context):
    """Genera la respuesta 'ground truth' usando solo el contexto."""
    ctx = "\n".join(context)

    prompt = f"""
Genera la respuesta correcta usando SOLO el CONTEXTO.
No inventes información externa.

### CONTEXTO:
{ctx}

### PREGUNTA:
{question}

### RESPUESTA CORRECTA:
"""
    return generate_message(client, prompt)


def main():
    print("=== Evaluación RAGAS — ChromaDB + Gemini ===\n")

    # Inicializamos Gemini REST (para respuestas y referencias)
    gemini_client = init_gemini_raw()

    # Inicializamos LLM para RAGAS (asíncrono)
    llm_for_ragas = GeminiLLM(GEMINI_MODEL)

    # Embeddings HuggingFace para RAGAS
    embedder = HFEmbedder()

    # Cargar colecciones
    col_text, col_img = get_collections()

    # Preguntas de prueba
    queries = [
        "¿Por qué es importante dormir bien?",
        "¿Qué beneficios aporta caminar 30 minutos diarios?",
        "Sugiere snacks saludables.",
        "¿Cuánta agua se recomienda tomar al día?",
        "¿Qué hábitos mejoran la calidad del sueño?"
    ]

    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truths": []
    }

    print("Generando dataset...\n")

    for q in queries:
        ctx = retrieve_context(q, col_text, col_img)

        answer = rag_answer(gemini_client, q, ctx)
        truth = generate_reference_answer(gemini_client, q, ctx)

        dataset_dict["question"].append(q)
        dataset_dict["answer"].append(answer)
        dataset_dict["contexts"].append(ctx)
        dataset_dict["ground_truths"].append([truth])

    dataset = Dataset.from_dict(dataset_dict)

    print("Ejecutando métricas RAGAS... Esto puede tardar.\n")

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm_for_ragas,
        embeddings=embedder
    )

    print("\n===== RESULTADOS RAGAS =====")
    print(result)

    df = result.to_pandas()
    df.to_csv("ragas_results.csv", index=False)

    print("\n✔ Resultados guardados en ragas_results.csv\n")


if __name__ == "__main__":
    main()
