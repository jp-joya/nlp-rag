import os
import sys
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

from ragas import Dataset, experiment
from ragas.metrics import DiscreteMetric

# Add parent directory to path to import rag_gemini module
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_gemini import rag_answer as rag_answer_chroma, init_gemini
from neo4j.rag_neo4j import rag_answer as rag_answer_neo4j

# Initialize Gemini
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Falta GOOGLE_API_KEY en .env")
genai.configure(api_key=api_key)

# Initialize the Gemini model directly (will be used for scoring)
gemini_model = init_gemini()
RAG_BACKEND = os.getenv("RAG_BACKEND", "chroma").lower()


def load_dataset():
    """Load nutrition-related test dataset"""
    dataset = Dataset(
        name="nutrition_rag_dataset",
        backend="local/csv",
        root_dir=".",
    )

    # Nutrition-specific test questions in Spanish
  # Nutrition-specific test questions based on your documents
    data_samples = [
        {
            "question": "¿Cuáles son las etapas principales del proceso digestivo y cuánto tiempo toma cada una?",
            "grading_notes": "- boca (0-30 segundos) con masticación - esófago (5-10 segundos) - estómago (1-4 horas) descomposición - intestino delgado (3-6 horas) absorción principal - intestino grueso (12-48 horas) - tiempo total varía según tipo de comida",
        },
        {
            "question": "¿Qué funciones tiene el microbioma intestinal?",
            "grading_notes": "- 70% del sistema inmunológico - producción de vitaminas K y B12 - produce 90% de serotonina - afecta salud mental y estado de ánimo - regula metabolismo y peso - protege barrera intestinal",
        },
        {
            "question": "¿Qué es el ayuno intermitente 16/8 y cuáles son sus beneficios?",
            "grading_notes": "- ayunas 16 horas y comes en ventana de 8 horas - mejora sensibilidad a insulina - promueve pérdida de peso - activa autofagia (limpieza celular) - mejora concentración - reduce inflamación sistémica",
        },
        {
            "question": "¿Cómo se debe planificar una comida equilibrada?",
            "grading_notes": "- proteína 150-200g (pollo, pescado, legumbres) - vegetales 2+ tazas variadas - carbohidrato complejo (arroz, papa, pasta) - grasas saludables (aceite de oliva, frutos secos) - incluir variedad de colores",
        },
        {
            "question": "¿Cuáles son las funciones de la vitamina D y dónde se encuentra?",
            "grading_notes": "- absorción de calcio y salud ósea - función inmunológica - regulación hormonal - salud mental - fuentes: luz solar 15-30 minutos, pescados grasos, huevos, lácteos enriquecidos",
        },
        {
            "question": "¿Qué factores dañan el microbioma intestinal?",
            "grading_notes": "- antibióticos destruyen bacterias buenas - pesticidas en alimentos no orgánicos - estrés crónico - dieta ultraprocesada sin fibra - falta de sueño - alcohol excesivo - falta de ejercicio",
        },
        {
            "question": "¿Cuáles son los beneficios de la planificación de comidas?",
            "grading_notes": "- mejora nutrición y variedad - control de peso y porciones - ahorro de dinero - reduce desperdicio - menos estrés diario - mayor consistencia - hábitos sostenibles",
        },
        {
            "question": "¿Qué minerales son esenciales y para qué sirven?",
            "grading_notes": "- hierro: transporte de oxígeno - calcio: salud ósea y muscular - magnesio: relajación muscular y sueño - zinc: inmunidad y cicatrización - potasio: función cardíaca y presión arterial - fuentes variadas en alimentos integrales",
        },
    ]

    for sample in data_samples:
        row = {
            "question": sample["question"], 
            "grading_notes": sample["grading_notes"]
        }
        dataset.append(row)

    # Save dataset
    dataset.save()
    return dataset


def score_response(response_text, grading_notes):
    """Score a response against grading notes using Gemini"""
    prompt = f"""Eres un evaluador experto de sistemas RAG (Retrieval-Augmented Generation) de nutrición.

    Tu tarea es evaluar si la respuesta generada es SATISFACTORIA considerando:
    1. ¿Está basada en el contexto proporcionado? (no debe inventar información)
    2. ¿Cubre los puntos principales de las notas de calificación?
    3. ¿Es precisa y útil para el usuario?

    RESPUESTA GENERADA:
    {response_text}

    PUNTOS CLAVE QUE DEBERÍA CUBRIR (Notas de Calificación):
    {grading_notes}

    CRITERIOS DE EVALUACIÓN:
    - PASS: La respuesta cubre al menos 60-70% de los puntos clave, está basada en el contexto (no inventa), y es útil
    - PASS: Puede usar palabras diferentes pero el significado es correcto
    - PASS: No necesita cubrir TODOS los puntos, pero sí los más importantes
    - FAIL: Inventa información que no está en el contexto
    - FAIL: Ignora completamente los puntos principales
    - FAIL: Respuesta muy vaga o irrelevante
    - FAIL: Dice que no tiene información cuando el contexto SÍ la contiene

    IMPORTANTE: Sé RAZONABLE. Si la respuesta es útil y mayormente correcta, aunque no sea perfecta, marca como PASS.

    Responde ÚNICAMENTE con una de estas dos palabras: pass o fail"""
    
    try:
        response = gemini_model.generate_content(prompt)
        result = response.text.strip().lower()
        # Extract pass or fail from response
        if "pass" in result:
            return "pass"
        elif "fail" in result:
            return "fail"
        else:
            return "fail"
    except Exception as e:
        print(f"Error scoring: {str(e)}")
        return "fail"


def generate_rag_response(question: str):
    """
    Ejecuta el backend de RAG seleccionado (Chroma/Gemini o Neo4j/Gemini)
    y normaliza la estructura de salida para el evaluador.
    """
    if RAG_BACKEND == "neo4j":
        response = rag_answer_neo4j(question)
        hits = response.get("context", [])
        text_ctx = "\n\n".join([h.get("content", "") for h in hits if h.get("type") == "text"])
        img_ctx = "\n\n".join([h.get("content", "") for h in hits if h.get("type") == "image_caption"])
        return {
            "answer": response.get("answer", ""),
            "context_text": text_ctx,
            "context_images": img_ctx,
        }

    # Default: Chroma + Gemini
    response = rag_answer_chroma(question)
    return {
        "answer": response.get("answer", ""),
        "context_text": response.get("context_text", ""),
        "context_images": response.get("context_images", ""),
    }


@experiment()
async def run_nutrition_experiment(row):
    """Run experiment with nutrition RAG system"""
    try:
        # Use your Gemini-based RAG system
        response = generate_rag_response(row["question"])

        # Score the response
        score = score_response(
            response.get("answer", ""),
            row["grading_notes"],
        )

        # Prepare experiment view with all relevant data
        experiment_view = {
            **row,
            "response": response.get("answer", ""),
            "score": score,
            "context_text": response.get("context_text", "")[:500],  # Truncate for readability
            "context_images": response.get("context_images", "")[:500],
            "backend": RAG_BACKEND,
        }
        return experiment_view
    
    except Exception as e:
        print(f"Error processing question: {row['question']}")
        print(f"Error: {str(e)}")
        return {
            **row,
            "response": f"ERROR: {str(e)}",
            "score": "fail",
            "context_text": "",
            "context_images": "",
        }


async def main():
    print("=== Iniciando Evaluación del Sistema RAG de Nutrición ===\n")
    print(f"Backend seleccionado: {RAG_BACKEND}\n")
    
    # Load dataset
    print("Cargando dataset de prueba...")
    dataset = load_dataset()
    print(f"Dataset cargado exitosamente: {len(dataset)} preguntas\n")
    
    # Run experiment
    print("Ejecutando experimento...")
    experiment_results = await run_nutrition_experiment.arun(dataset)
    print("\n¡Experimento completado exitosamente!")
    
    # Display summary
    print("\n=== Resumen de Resultados ===")
    total = len(experiment_results)
    passed = sum(1 for r in experiment_results if r.get("score") == "pass")
    print(f"Total de preguntas: {total}")
    print(f"Aprobadas: {passed}")
    print(f"Fallidas: {total - passed}")
    print(f"Tasa de éxito: {(passed/total)*100:.1f}%")
    
    # Save experiment results
    experiment_results.save()
    csv_path = Path(".") / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nResultados guardados en: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
