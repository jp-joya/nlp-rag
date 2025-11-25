# 🏗️ RAGAS Integration Architecture

## Sistema de Evaluación de RAG - Arquitectura Completa

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ENTRADA DEL USUARIO                             │
│                    (evaluate_all.py)                                 │
│              Script Principal - Controlador Maestro                  │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
        ┌──────────┼──────────┬──────────────┬──────────────┐
        │          │          │              │              │
        ▼          ▼          ▼              ▼              ▼
    ┌─────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  RAGAS  │ │Dashboard│ │  Quick   │ │ Reporte │ │  Ayuda   │
    │ Completa│ │ Métricas│ │ Quality  │ │  HTML   │ │  (Help)  │
    └────┬────┘ └────┬───┘ └──────┬───┘ └──────┬───┘ └──────────┘
         │           │           │            │
         ▼           ▼           ▼            ▼
    ┌──────────────────────────────────────────────────────┐
    │           CAPA DE EVALUACIÓN                         │
    │                                                      │
    │  ┌────────────────────────────────────────────┐    │
    │  │    ragas_evaluator.py                     │    │
    │  ├────────────────────────────────────────────┤    │
    │  │ • Generate Test Dataset                  │    │
    │  │ • Evaluate with RAGAS Metrics            │    │
    │  │ • Save Results (JSON/CSV)                │    │
    │  │ • Print Visual Summary                   │    │
    │  └────────────────────────────────────────────┘    │
    │                                                      │
    │  ┌────────────────────────────────────────────┐    │
    │  │    rag_metrics_integration.py              │    │
    │  ├────────────────────────────────────────────┤    │
    │  │ • @measure_rag_performance (decorator)    │    │
    │  │ • quick_quality_check()                   │    │
    │  │ • RAGMetricsCollector                      │    │
    │  │ • RAGQualityEvaluator                      │    │
    │  └────────────────────────────────────────────┘    │
    │                                                      │
    │  ┌────────────────────────────────────────────┐    │
    │  │    metrics_dashboard.py                    │    │
    │  ├────────────────────────────────────────────┤    │
    │  │ • Display Historical Metrics               │    │
    │  │ • Print Quick Status                       │    │
    │  │ • Export Reports (CSV/HTML)                │    │
    │  │ • Track Trends                             │    │
    │  └────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌───────────┐
    │  RAGAS   │  │ LLM      │  │ Embedder  │
    │ Metrics  │  │ (Gemini) │  │ (HuggingF)│
    └──────────┘  └──────────┘  └───────────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
    ┌─────────────────┐        ┌─────────────────┐
    │   ChromaDB      │        │   rag_answer()  │
    │   Collections   │        │   RAG System    │
    │ • Text docs     │        │                 │
    │ • Image captions│        └─────────────────┘
    └─────────────────┘
```

---

## 📊 Flujo de Datos - Evaluación RAGAS Completa

```
INPUT: query
   │
   ├─► RAG System (rag_answer)
   │       │
   │       ├─► Retrieve Context (ChromaDB)
   │       │   ├─ Text chunks
   │       │   └─ Image descriptions
   │       │
   │       └─► Generate Response (Gemini)
   │           └─ Output: answer
   │
   ├─► RAGAS Framework
   │       │
   │       ├─► Faithfulness Metric
   │       │   └─ ¿Answer based on context?
   │       │
   │       ├─► Answer Relevancy
   │       │   └─ ¿Answer relevant to query?
   │       │
   │       ├─► Context Relevancy
   │       │   └─ ¿Context relevant to query?
   │       │
   │       └─► Context Precision
   │           └─ ¿Context precise & clean?
   │
   └─► OUTPUT: Scores (0-1) for each metric
```

---

## 🎯 Tipos de Evaluación

### 1. Evaluación Completa RAGAS
```
evaluate_all.py full
    │
    ├─► ragas_evaluator.py
    │   • 10 test queries
    │   • RAGAS evaluation
    │   • Detailed metrics
    │   • Time: 5-15 min
    │
    ├─► metrics_dashboard.py
    │   • Historical trends
    │   • Quality status
    │   • Time: <1 min
    │
    └─► Generate Report
        • HTML report
        • CSV export
        • Time: <1 min
```

### 2. Verificación Rápida
```
evaluate_all.py quick
    │
    ├─► 3 test queries
    ├─► quick_quality_check()
    │   • Context length
    │   • Response length
    │   • Response diversity
    │   • Context relevance (proxy)
    │
    └─► Overall Score
        • Time: <1 min
```

### 3. Monitoreo Continuo
```
metrics_dashboard.py
    │
    ├─► Load metrics_history.csv
    ├─► Display trends
    ├─► Show health status
    │
    └─► Optional: Export reports
        • Time: <1 min
```

---

## 💾 Almacenamiento de Datos

### Estructura de Directorios
```
nlp-rag/
├── ragas_results.json           # Latest RAGAS results
├── ragas_results.csv            # Latest RAGAS results (table)
├── session_metrics.json         # Current session stats
├── quick_quality_check.json     # Quick eval results
│
├── ragas_evaluations/
│   ├── latest_results.json      # Last evaluation
│   ├── metrics_history.csv      # All past evaluations
│   │
│   └── reports/                 # (Optional)
│       ├── metrics_report_*.csv
│       └── metrics_report_*.html
│
└── [other files...]
```

### Datos Almacenados

```json
{
  "timestamp": "2024-11-24T22:30:00",
  "total_questions": 10,
  "metrics": {
    "faithfulness": {
      "mean": 0.8523,
      "min": 0.7100,
      "max": 0.9200,
      "scores": [0.85, 0.82, 0.90, ...]
    },
    "answer_relevancy": {...},
    "context_relevancy": {...},
    "context_precision": {...}
  },
  "detailed_results": [
    {
      "id": 0,
      "question": "¿Por qué es importante dormir bien?",
      "answer": "Dormir bien...",
      "faithfulness": 0.85,
      "answer_relevancy": 0.78,
      ...
    },
    ...
  ]
}
```

---

## 🔄 Ciclo de Mejora Continua

```
PASO 1: Evalúa
├─► python evaluate_all.py full
├─► Genera metrics baseline
└─► Archiva resultados

PASO 2: Analiza
├─► python metrics_dashboard.py status
├─► Identifica métricas bajas
└─► Prioriza mejoras

PASO 3: Mejora
├─► Actualiza prompts
├─► Mejora embeddings
├─► Optimiza retrieval
└─► Ajusta parámetros

PASO 4: Re-Evalúa
├─► python evaluate_all.py ragas
├─► Compara con baseline
└─► Valida mejoras

PASO 5: Monitor
├─► python metrics_dashboard.py dashboard
├─► Observa tendencias
└─► Detecta regresiones

REPETIR: Ir a PASO 2
```

---

## 📈 Métricas por Componente

### RAGAS Metrics (Evaluación Profunda)
```
┌─ Faithfulness (0-1)
│  • ¿Respuesta basada en contexto?
│  • Detecta alucinaciones
│  • Requiere: LLM para evaluación
│
├─ Answer Relevancy (0-1)
│  • ¿Respuesta relevante a pregunta?
│  • Valida utilidad de la respuesta
│  • Requiere: LLM para evaluación
│
├─ Context Relevancy (0-1)
│  • ¿Contexto relevante?
│  • Valida retriever
│  • Requiere: LLM + Embeddings
│
└─ Context Precision (0-1)
   • ¿Contexto limpio y preciso?
   • Valida calidad de chunks
   • Requiere: LLM evaluation
```

### Quick Metrics (Evaluación Rápida)
```
┌─ Context Length
│  • ¿Contexto suficientemente largo?
│  • Rápido: Sin LLM
│
├─ Response Length
│  • ¿Respuesta suficientemente completa?
│  • Rápido: Sin LLM
│
├─ Response Diversity
│  • ¿Variedad de vocabulario?
│  • Rápido: Sin LLM
│
└─ Context Relevance (Proxy)
   • ¿Overlap de términos?
   • Rápido: Sin LLM
```

### Performance Metrics (Integración)
```
┌─ Latency
│  * Tiempo total de respuesta
│  * Min/Max/Average
│
├─ Throughput
│  * Queries por segundo
│  * Rate de procesamiento
│
├─ Error Rate
│  * Porcentaje de fallos
│  * Tipos de error
│
└─ Resource Usage
   * Memoria
   * CPU
   * API calls
```

---

## 🚀 Integración en Producción

```python
# Opción 1: En tu app Flask (app.py)
from rag_metrics_integration import measure_rag_performance

@app.route('/api/query', methods=['POST'])
@measure_rag_performance
def query():
    # Tu código aquí
    return result

# Opción 2: Monitoring periódico
import schedule

def evaluate_periodically():
    subprocess.run(['python', 'ragas_evaluator.py'])

schedule.every().week.do(evaluate_periodically)

# Opción 3: Webhook alertas
def alert_if_metrics_degrade():
    status = get_metrics_status()
    if status['faithfulness']['score'] < 0.7:
        send_alert("Faithfulness dropped below 0.7!")
```

---

## 📊 Dashboard Típico

```
═══════════════════════════════════════════════════════════════════
⚡ QUICK METRICS STATUS
═══════════════════════════════════════════════════════════════════
📅 Última actualización: 2024-11-24 22:30:00
❓ Preguntas evaluadas: 10

🟢 Faithfulness: 0.8523
   Health: EXCELENTE (0.8+)

🟡 Answer Relevancy: 0.6845
   Health: BUENO (0.6+)

🟢 Context Relevancy: 0.7892
   Health: BUENO (0.6+)

🟢 Context Precision: 0.8234
   Health: EXCELENTE (0.8+)
═══════════════════════════════════════════════════════════════════

📈 TENDENCIA: ↑ MEJORA (Last 5 vs First 5)
```

---

## ✅ Checklist de Implementación

- [x] Crear ragas_evaluator.py
- [x] Crear metrics_dashboard.py
- [x] Crear rag_metrics_integration.py
- [x] Crear evaluate_all.py (controlador)
- [x] Crear RAGAS_GUIDE.md (documentación)
- [x] Crear RAGAS_SETUP.md (resumen)
- [x] Crear requirements-ragas.txt
- [x] Documentar arquitectura

**¡Sistema de evaluación listo para usar! 🎉**
