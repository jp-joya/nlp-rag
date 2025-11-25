"""
Integration Layer - Integra RAGAS evaluations en el flujo RAG
Proporciona decoradores y utilidades para evaluación automática
"""

import functools
import json
import time
from datetime import datetime
from typing import Callable, Any, Dict


class RAGMetricsCollector:
    """Colecta métricas durante la ejecución del RAG"""

    def __init__(self):
        self.queries = []
        self.responses = []
        self.contexts = []
        self.latencies = []
        self.errors = []

    def log_query(
        self,
        query: str,
        response: str,
        context: str,
        latency: float,
        error: str = None,
    ):
        """Registra una consulta con su respuesta"""
        self.queries.append(query)
        self.responses.append(response)
        self.contexts.append(context)
        self.latencies.append(latency)

        if error:
            self.errors.append(
                {
                    "query": query,
                    "error": error,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de desempeño"""
        if not self.latencies:
            return {}

        return {
            "total_queries": len(self.queries),
            "average_latency": sum(self.latencies) / len(self.latencies),
            "min_latency": min(self.latencies),
            "max_latency": max(self.latencies),
            "total_errors": len(self.errors),
            "timestamp": datetime.now().isoformat(),
        }

    def save_session(self, filename: str = "session_metrics.json"):
        """Guarda métricas de la sesión"""
        data = {
            "metadata": self.get_stats(),
            "queries": self.queries,
            "responses": [r[:500] + "..." if len(r) > 500 else r for r in self.responses],
            "contexts": [c[:300] + "..." if len(c) > 300 else c for c in self.contexts],
            "errors": self.errors,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"📊 Sesión guardada en: {filename}")


# Instancia global
_metrics_collector = RAGMetricsCollector()


def measure_rag_performance(func: Callable) -> Callable:
    """
    Decorador que mide performance de funciones RAG
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            latency = time.time() - start_time

            # Log exitoso
            query = args[0] if args else kwargs.get("query", "unknown")
            if isinstance(result, dict):
                response = result.get("answer", "")
                context = result.get("context_text", "")
            else:
                response = str(result)
                context = ""

            _metrics_collector.log_query(query, response, context, latency)

            print(
                f"✅ Query completada en {latency:.2f}s | Función: {func.__name__}"
            )

            return result

        except Exception as e:
            latency = time.time() - start_time
            query = args[0] if args else kwargs.get("query", "unknown")

            _metrics_collector.log_query(
                query, "", "", latency, error=str(e)
            )

            print(
                f"❌ Error en {func.__name__} después de {latency:.2f}s: {str(e)}"
            )

            raise

    return wrapper


def get_session_stats() -> Dict[str, Any]:
    """Obtiene estadísticas de la sesión actual"""
    return _metrics_collector.get_stats()


def save_session_metrics(filename: str = "session_metrics.json"):
    """Guarda métricas de la sesión"""
    _metrics_collector.save_session(filename)


def get_metrics_collector() -> RAGMetricsCollector:
    """Retorna el colector de métricas global"""
    return _metrics_collector


# ======================== QUALITY METRICS ========================


class RAGQualityEvaluator:
    """Evalúa calidad del RAG sin necesidad de RAGAS"""

    @staticmethod
    def evaluate_context_length(context: str, min_length: int = 100) -> Dict:
        """Evalúa si el contexto es suficientemente largo"""
        return {
            "metric": "context_length",
            "value": len(context),
            "min_threshold": min_length,
            "pass": len(context) >= min_length,
            "score": min(len(context) / min_length, 1.0),
        }

    @staticmethod
    def evaluate_response_length(response: str, min_length: int = 50) -> Dict:
        """Evalúa si la respuesta es suficientemente completa"""
        return {
            "metric": "response_length",
            "value": len(response),
            "min_threshold": min_length,
            "pass": len(response) >= min_length,
            "score": min(len(response) / min_length, 1.0),
        }

    @staticmethod
    def evaluate_response_diversity(response: str) -> Dict:
        """Evalúa la diversidad de palabras en la respuesta"""
        words = response.lower().split()
        unique_words = len(set(words))
        diversity_ratio = unique_words / len(words) if words else 0

        return {
            "metric": "response_diversity",
            "unique_words": unique_words,
            "total_words": len(words),
            "score": diversity_ratio,
        }

    @staticmethod
    def evaluate_context_relevance_proxy(query: str, context: str) -> Dict:
        """Evaluación simple de relevancia (sin LLM)"""
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())

        overlap = len(query_words & context_words)
        relevance_score = overlap / len(query_words) if query_words else 0

        return {
            "metric": "context_relevance_proxy",
            "overlapping_terms": overlap,
            "score": min(relevance_score, 1.0),
        }


def quick_quality_check(query: str, context: str, answer: str) -> Dict:
    """Realiza un chequeo rápido de calidad sin RAGAS"""
    evaluator = RAGQualityEvaluator()

    results = {
        "timestamp": datetime.now().isoformat(),
        "query_length": len(query),
        "context_length_check": evaluator.evaluate_context_length(context),
        "response_length_check": evaluator.evaluate_response_length(answer),
        "response_diversity_check": evaluator.evaluate_response_diversity(answer),
        "context_relevance_check": evaluator.evaluate_context_relevance_proxy(
            query, context
        ),
    }

    # Calcular score general
    scores = [
        results["context_length_check"]["score"],
        results["response_length_check"]["score"],
        results["response_diversity_check"]["score"],
        results["context_relevance_check"]["score"],
    ]

    results["overall_quality_score"] = sum(scores) / len(scores) if scores else 0

    return results


def print_quality_report(quality_check: Dict):
    """Imprime reporte visual de calidad"""
    print("\n" + "=" * 70)
    print("🔍 QUICK QUALITY ASSESSMENT")
    print("=" * 70)

    score = quality_check["overall_quality_score"]

    if score >= 0.8:
        emoji = "🟢 EXCELENTE"
    elif score >= 0.6:
        emoji = "🟡 BUENO"
    else:
        emoji = "🔴 NECESITA MEJORA"

    print(f"\n{emoji} Puntuación general: {score:.4f}\n")

    print("Detalles:")
    print(
        f"  Context Length: {quality_check['context_length_check']['value']} chars "
        f"({quality_check['context_length_check']['score']:.2%})"
    )
    print(
        f"  Response Length: {quality_check['response_length_check']['value']} chars "
        f"({quality_check['response_length_check']['score']:.2%})"
    )
    print(
        f"  Response Diversity: {quality_check['response_diversity_check']['unique_words']} "
        f"palabras únicas ({quality_check['response_diversity_check']['score']:.2%})"
    )
    print(
        f"  Context Relevance: "
        f"({quality_check['context_relevance_check']['score']:.2%})"
    )

    print("\n" + "=" * 70 + "\n")


# ======================== EXAMPLE USAGE ========================


def main():
    """Ejemplo de uso de métricas"""
    from rag_gemini import rag_answer

    @measure_rag_performance
    def evaluate_query(query: str):
        return rag_answer(query)

    # Ejemplo
    print("🚀 Testing RAG Metrics Integration\n")

    test_queries = [
        "¿Por qué es importante dormir?",
        "¿Qué son los macronutrientes?",
    ]

    for query in test_queries:
        try:
            result = evaluate_query(query)

            # Quick quality check
            quality = quick_quality_check(
                query, result["context_text"], result["answer"]
            )
            print_quality_report(quality)

        except Exception as e:
            print(f"❌ Error: {e}")

    # Guardar sesión
    print("\n📊 Guardando métricas de sesión...")
    save_session_metrics()

    print("\n📈 Estadísticas de sesión:")
    stats = get_session_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
