import os
from typing import List, Dict

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ============================
# CONFIGURACIÓN
# ============================

# Como estás ejecutando el script en la misma EC2 que Neo4j,
# podemos usar localhost. Si quieres, puedes cambiar a la IP pública.
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "adminadmin"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# El enunciado pide Gemini Flash 2.5.
# En la API actual esto se expone como gemini-2.0-flash (modelo flash de última generación).
GEMINI_MODEL = "gemini-2.0-flash"


# ============================
# EMBEDDINGS Y NEO4J
# ============================

embedder = SentenceTransformer(EMBED_MODEL)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def embed_query(query: str):
    """Genera embedding usando SentenceTransformer."""
    return embedder.encode(query).tolist()


def retrieve_from_neo4j(query: str, top_k: int = 5) -> List[Dict]:
    """
    Realiza búsqueda vectorial en Neo4j usando cosine similarity manual
    entre el embedding de la consulta y los nodos :Embedding.

    Retorna una lista de diccionarios:
    {
      "type": "text" | "image_caption",
      "score": float,
      "content": str
    }
    """
    vector = embed_query(query)

    cypher = """
    WITH $query_vector AS query
    MATCH (e:Embedding)

    WITH e, query,
         reduce(dot = 0.0, i IN range(0, size(query)-1) |
             dot + query[i] * e.vector[i]
         ) AS dot,
         reduce(mq = 0.0, i IN range(0, size(query)-1) |
             mq + query[i]^2
         ) AS mag_query,
         reduce(me = 0.0, i IN range(0, size(e.vector)-1) |
             me + e.vector[i]^2
         ) AS mag_emb

    WITH e,
         dot / (sqrt(mag_query) * sqrt(mag_emb)) AS score
    ORDER BY score DESC
    LIMIT $k

    MATCH (owner)-[:HAS_EMBEDDING]->(e)
    RETURN owner, score;
    """

    with driver.session() as session:
        result = session.run(cypher, query_vector=vector, k=top_k)

        outputs: List[Dict] = []
        for record in result:
            owner = record["owner"]
            score = record["score"]

            if "Chunk" in owner.labels:
                outputs.append({
                    "type": "text",
                    "score": score,
                    "content": owner["text"],
                })
            elif "Caption" in owner.labels:
                outputs.append({
                    "type": "image_caption",
                    "score": score,
                    "content": owner["text"],
                })

        return outputs


# ============================
# GEMINI
# ============================

def init_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "La variable de entorno GOOGLE_API_KEY no está configurada. "
            "Configúrala con tu API key de Google AI Studio."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def build_context_blocks(hits: List[Dict]):
    """
    Separa el contexto en dos bloques:
    - contexto textual
    - contexto de imágenes (descripciones / captions)
    """
    text_snippets = []
    image_snippets = []

    for h in hits:
        if h["type"] == "text":
            text_snippets.append(h["content"])
        elif h["type"] == "image_caption":
            image_snippets.append(h["content"])

    return text_snippets, image_snippets


def build_prompt(question: str, text_snippets: List[str], image_snippets: List[str]) -> str:
    """
    Construye el prompt que se enviará a Gemini,
    incluyendo contexto textual y descripciones de imágenes.
    """
    ctx_text_block = "\n\n".join(f"- {t}" for t in text_snippets) if text_snippets else "N/A"
    ctx_img_block = "\n\n".join(f"- {c}" for c in image_snippets) if image_snippets else "N/A"

    prompt = f"""
Eres un asistente de apoyo en un sistema RAG para un proyecto de nutrición y hábitos saludables.
Debes responder en español, de forma clara, breve y basada EXCLUSIVAMENTE en el contexto proporcionado.

Pregunta del usuario:
\"\"\" 
{question}
\"\"\"

--- CONTEXTO TEXTO (fragmentos de documentos) ---
{ctx_text_block}

--- CONTEXTO IMÁGENES (descripciones/captions) ---
{ctx_img_block}

Instrucciones:
- Usa principalmente el contexto textual.
- Si alguna descripción de imagen aporta algo relevante (por ejemplo, frutas, ensalada, hombre durmiendo, caminata, agua, etc.), menciónala brevemente como ejemplo.
- Si el contexto no fuera suficiente para responder, dilo explícitamente y no inventes información.
- Responde en uno o dos párrafos máximo, salvo que la pregunta requiera más detalle.
"""
    return prompt


def answer_with_gemini(model, question: str, hits: List[Dict]) -> str:
    text_snippets, image_snippets = build_context_blocks(hits)
    prompt = build_prompt(question, text_snippets, image_snippets)
    response = model.generate_content(prompt)
    return response.text


def rag_answer(question: str) -> Dict[str, object]:
    """
    Genera una respuesta usando Neo4j como base de conocimiento y Gemini como modelo generativo.
    Recibe únicamente la pregunta del usuario.
    """
    hits = retrieve_from_neo4j(question, top_k=5)
    model = init_gemini()
    answer = answer_with_gemini(model, question, hits)

    return {
        "question": question,
        "context": hits,
        "answer": answer,
    }


# ============================
# MAIN DEMO
# ============================

def main():
    print("\n=== RAG con Neo4j + Gemini (Nutrición) ===\n")

    question = input("Escribe tu pregunta sobre nutrición / hábitos saludables:\n> ").strip()
    if not question:
        print("No se ingresó pregunta. Saliendo.")
        return

    print("\n[1] Recuperando contexto desde Neo4j...")
    hits = retrieve_from_neo4j(question, top_k=5)

    print("\n--- CONTEXTO RECUPERADO ---")
    for i, h in enumerate(hits):
        print(f"\n[{i}] Tipo: {h['type']} | score={h['score']:.4f}")
        print(h["content"])

    print("\n[2] Inicializando Gemini...")
    model = init_gemini()

    print("\n[3] Generando respuesta...")
    answer = answer_with_gemini(model, question, hits)

    print("\n===== RESPUESTA GEMINI (Neo4j) =====\n")
    print(answer)
    print("\n====================================\n")


if __name__ == "__main__":
    main()
