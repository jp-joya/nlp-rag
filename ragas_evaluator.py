"""
RAGAS Evaluation Framework for RAG System
Evaluates: Faithfulness, Answer Relevancy, Context Relevancy, Context Precision
"""

import os
import json
import chromadb
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Dataset
from datasets import Dataset

# RAGAS Metrics
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_precision,
)

# Custom imports
from gemini_llm import GeminiLLM
from hf_embedder import HFEmbedder
from rag_gemini import rag_answer

# ===================== CONFIG =====================
CHROMA_PATH = "./chroma_db"
TEXT_COLLECTION = "nutricion_textos"
IMAGE_COLLECTION = "nutricion_imagenes"
RESULTS_FILE = "./ragas_results.json"
RESULTS_CSV = "./ragas_results.csv"

# Test Questions
TEST_QUERIES = [
    "¿Por qué es importante dormir bien?",
    "¿Cuáles son los macronutrientes principales?",
    "¿Cuánta agua debo beber diariamente?",
    "¿Qué beneficios tiene caminar 30 minutos?",
    "¿Qué son los snacks saludables?",
    "¿Cómo manejo el estrés?",
    "¿Qué son las vitaminas y minerales?",
    "¿Cómo funciona el ayuno intermitente?",
    "¿Cómo planificar comidas saludables?",
    "¿Qué es la salud digestiva?",
]

# ================================================


def init_gemini():
    """Inicializa Gemini LLM con API key"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY en .env")
    return GeminiLLM(api_key=api_key)


def init_embedder():
    """Inicializa embedder de HuggingFace"""
    return HFEmbedder()


def get_chroma_collections():
    """Obtiene colecciones de Chroma"""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col_text = client.get_collection(TEXT_COLLECTION)
    col_images = client.get_collection(IMAGE_COLLECTION)
    return col_text, col_images


def retrieve_context_for_eval(query: str, col_text, col_images, k_text=1, k_img=1) -> Tuple[str, str]:
    """
    Recupera contexto para evaluación
    Retorna: (context_text, context_images)
    """
    # Búsqueda en textos
    res_text = col_text.query(query_texts=[query], n_results=k_text)
    text_chunks = res_text.get("documents", [[]])[0]
    text_metas = res_text.get("metadatas", [[]])[0]

    ctx_text = ""
    for doc, meta in zip(text_chunks, text_metas):
        src = meta.get("source", "desconocido")
        ctx_text += f"[Fuente: {src}]\n{doc}\n\n"

    # Búsqueda en imágenes
    res_img = col_images.query(query_texts=[query], n_results=k_img)
    img_chunks = res_img.get("documents", [[]])[0]
    img_metas = res_img.get("metadatas", [[]])[0]

    ctx_img = ""
    for caption, meta in zip(img_chunks, img_metas):
        img_path = meta.get("source_image", "desconocida")
        ctx_img += f"[Imagen: {img_path}]\nDescripción: {caption}\n\n"

    return ctx_text.strip(), ctx_img.strip()


def generate_test_dataset(llm, embedder, col_text, col_images) -> Dataset:
    """
    Genera dataset de prueba con queries, ground truth y contexto
    """
    data = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": [],
    }

    print("📊 Generando dataset de evaluación...")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"  [{i}/{len(TEST_QUERIES)}] Procesando: {query[:50]}...")

        try:
            # Obtén respuesta RAG
            result = rag_answer(query)
            answer = result["answer"]
            context_text = result["context_text"]

            # Contexto como lista (RAGAS espera lista)
            contexts = [context_text] if context_text else [""]

            # Ground truth: usa el mismo answer (idealmente tendrías verdad terreno)
            ground_truth = answer

            data["question"].append(query)
            data["contexts"].append(contexts)
            data["answer"].append(answer)
            data["ground_truth"].append(ground_truth)

        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
            continue

    print(f"✅ Dataset generado con {len(data['question'])} ejemplos\n")

    return Dataset.from_dict(data)


def evaluate_rag(dataset: Dataset, llm: GeminiLLM, embedder: HFEmbedder) -> Dict:
    """
    Evalúa RAG usando RAGAS con múltiples métricas
    """
    print("🔍 Evaluando RAG con RAGAS...\n")

    try:
        # Evalúa con múltiples métricas
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_relevancy,
                context_precision,
            ],
            llm=llm,
            embeddings=embedder,
        )

        return results

    except Exception as e:
        print(f"❌ Error durante evaluación: {str(e)}")
        return None


def save_results(results, dataset: Dataset):
    """
    Guarda resultados en JSON y CSV
    """
    if results is None:
        print("❌ No hay resultados para guardar")
        return

    print("💾 Guardando resultados...\n")

    # Preparar datos para guardar
    metrics_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(dataset),
        "metrics": {},
        "detailed_results": [],
    }

    # Extraer métricas agregadas
    if hasattr(results, "scores"):
        for metric_name in results.scores.keys():
            scores = results.scores[metric_name]
            if isinstance(scores, list):
                metrics_summary["metrics"][metric_name] = {
                    "mean": float(sum(scores) / len(scores)),
                    "min": float(min(scores)),
                    "max": float(max(scores)),
                    "scores": scores,
                }

    # Resultados detallados
    for i, (question, answer, contexts) in enumerate(
        zip(
            dataset["question"],
            dataset["answer"],
            dataset["contexts"],
        )
    ):
        detail = {
            "id": i,
            "question": question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "context_used": contexts[0][:200] + "..." if contexts and len(contexts[0]) > 200 else (contexts[0] if contexts else ""),
        }

        # Agregar scores por métrica
        if hasattr(results, "scores"):
            for metric_name in results.scores.keys():
                if isinstance(results.scores[metric_name], list):
                    if i < len(results.scores[metric_name]):
                        detail[metric_name] = results.scores[metric_name][i]

        metrics_summary["detailed_results"].append(detail)

    # Guardar JSON
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Resultados guardados en: {RESULTS_FILE}")

    # Guardar CSV
    if metrics_summary["detailed_results"]:
        df = pd.DataFrame(metrics_summary["detailed_results"])
        df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
        print(f"✅ CSV guardado en: {RESULTS_CSV}")

    return metrics_summary


def print_summary(metrics_summary: Dict):
    """
    Imprime resumen visual de métricas
    """
    print("\n" + "=" * 70)
    print("📈 RAGAS EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\n📅 Evaluación: {metrics_summary['timestamp']}")
    print(f"❓ Total de preguntas: {metrics_summary['total_questions']}")

    print("\n" + "-" * 70)
    print("🎯 MÉTRICAS PRINCIPALES:")
    print("-" * 70)

    metrics = metrics_summary.get("metrics", {})

    if not metrics:
        print("❌ No hay métricas para mostrar")
        return

    for metric_name, values in metrics.items():
        if isinstance(values, dict) and "mean" in values:
            mean = values["mean"]
            min_val = values["min"]
            max_val = values["max"]

            # Emojis según rendimiento
            if mean >= 0.8:
                emoji = "🟢"
            elif mean >= 0.6:
                emoji = "🟡"
            else:
                emoji = "🔴"

            print(f"\n{emoji} {metric_name.upper()}")
            print(f"  Mean:  {mean:.4f}")
            print(f"  Min:   {min_val:.4f}")
            print(f"  Max:   {max_val:.4f}")

    print("\n" + "=" * 70)
    print("💡 INTERPRETACIÓN:")
    print("-" * 70)
    print("""
Faithfulness (0-1):       ¿La respuesta está basada en el contexto?
                          - 0.8+: Excelente
                          - 0.6-0.8: Bueno
                          - <0.6: Necesita mejora

Answer Relevancy (0-1):   ¿La respuesta es relevante a la pregunta?
                          - 0.8+: Muy relevante
                          - 0.6-0.8: Relevante
                          - <0.6: Poco relevante

Context Relevancy (0-1):  ¿El contexto recuperado es relevante?
                          - 0.8+: Contexto excelente
                          - 0.6-0.8: Contexto bueno
                          - <0.6: Contexto pobre

Context Precision (0-1):  ¿Qué tan preciso es el contexto?
                          - 0.8+: Muy preciso
                          - 0.6-0.8: Preciso
                          - <0.6: Impreciso
    """)
    print("=" * 70 + "\n")


def main():
    """Ejecuta evaluación RAGAS completa"""
    print("\n🚀 INICIANDO EVALUACIÓN RAGAS\n")

    try:
        # Inicializar componentes
        print("🔧 Inicializando componentes...")
        llm = init_gemini()
        embedder = init_embedder()
        col_text, col_images = get_chroma_collections()
        print("✅ Componentes inicializados\n")

        # Generar dataset
        dataset = generate_test_dataset(llm, embedder, col_text, col_images)

        if len(dataset) == 0:
            print("❌ Dataset vacío, abortando")
            return

        # Evaluar
        results = evaluate_rag(dataset, llm, embedder)

        # Guardar resultados
        metrics_summary = save_results(results, dataset)

        # Mostrar resumen
        if metrics_summary:
            print_summary(metrics_summary)
        else:
            print("❌ No se pudieron obtener resumen de métricas")

    except Exception as e:
        print(f"\n❌ ERROR FATAL: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
