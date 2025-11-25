# RAGAS Evaluation Framework - Documentación

## 📊 ¿Qué es RAGAS?

RAGAS (RAG Assessment) es un framework para evaluar sistemas de pregunta y respuesta basados en Retrieval-Augmented Generation. Proporciona métricas automáticas para medir la calidad de tu RAG.

## 🎯 Métricas Disponibles

### 1. **Faithfulness** (Fidelidad)
- **¿Qué mide?**: ¿La respuesta está basada en el contexto recuperado?
- **Rango**: 0 a 1 (donde 1 es mejor)
- **Interpretación**:
  - 0.8+: La respuesta es fiel al contexto
  - 0.6-0.8: La mayoría está basada en el contexto
  - <0.6: La respuesta tiene mucha información fuera del contexto

### 2. **Answer Relevancy** (Relevancia de Respuesta)
- **¿Qué mide?**: ¿Qué tan relevante es la respuesta a la pregunta?
- **Rango**: 0 a 1
- **Interpretación**:
  - 0.8+: Muy relevante y directa
  - 0.6-0.8: Relevante pero con información extra
  - <0.6: Poco relevante o no responde la pregunta

### 3. **Context Relevancy** (Relevancia del Contexto)
- **¿Qué mide?**: ¿Qué tan relevante es el contexto recuperado?
- **Rango**: 0 a 1
- **Interpretación**:
  - 0.8+: Contexto muy relevante
  - 0.6-0.8: Contexto útil pero con ruido
  - <0.6: Contexto poco relevante

### 4. **Context Precision** (Precisión del Contexto)
- **¿Qué mide?**: ¿Qué tan preciso y limpio es el contexto?
- **Rango**: 0 a 1
- **Interpretación**:
  - 0.8+: Contexto muy preciso
  - 0.6-0.8: Contexto con algunas partes innecesarias
  - <0.6: Mucho ruido en el contexto

## 🚀 Cómo Usar

### Opción 1: Evaluación Completa con RAGAS

```bash
# Ejecuta evaluación RAGAS
python ragas_evaluator.py
```

**Output esperado**:
- Generará dataset de prueba con 10 preguntas
- Evaluará cada metrica
- Guardará resultados en `ragas_results.json` y `ragas_results.csv`
- Mostrará resumen visual

**Tiempo estimado**: 5-15 minutos (depende de conexión a API)

### Opción 2: Monitoreo de Métricas

```bash
# Ver estado rápido
python metrics_dashboard.py status

# Ver histórico completo
python metrics_dashboard.py dashboard

# Exportar reporte
python metrics_dashboard.py export csv
python metrics_dashboard.py export html
```

### Opción 3: Integración en tu Código

```python
from rag_metrics_integration import (
    measure_rag_performance,
    quick_quality_check,
    print_quality_report,
    save_session_metrics
)
from rag_gemini import rag_answer

# Decorador para medir performance automáticamente
@measure_rag_performance
def my_query(query):
    return rag_answer(query)

# Usar
result = my_query("¿Qué son los macronutrientes?")

# Evaluación rápida sin RAGAS
quality = quick_quality_check(
    query=result['query'],
    context=result['context_text'],
    answer=result['answer']
)
print_quality_report(quality)

# Guardar métricas de sesión
save_session_metrics("my_session_metrics.json")
```

## 📈 Interpretando Resultados

### Ejemplo de Reporte RAGAS

```
===============================================================
📈 RAGAS EVALUATION SUMMARY
===============================================================

📅 Evaluación: 2024-11-24T22:30:00.123456
❓ Total de preguntas: 10

-----------  MÉTRICAS PRINCIPALES -----------

🟢 FAITHFULNESS
  Mean:  0.8523
  Min:   0.7100
  Max:   0.9200

🟡 ANSWER_RELEVANCY
  Mean:  0.6845
  Min:   0.5200
  Max:   0.8900

🟢 CONTEXT_RELEVANCY
  Mean:  0.7892
  Min:   0.6500
  Max:   0.9100

🟢 CONTEXT_PRECISION
  Mean:  0.8234
  Min:   0.7000
  Max:   0.9400
```

### Interpretación:
- **Faithfulness 0.85**: ¡Excelente! Las respuestas están basadas en contexto
- **Answer Relevancy 0.68**: Bien, pero hay mejora posible
- **Context Relevancy 0.79**: Bueno, recuperas contexto relevante
- **Context Precision 0.82**: Excelente precisión de contexto

## 🔧 Configuración

### Variables Modificables en `ragas_evaluator.py`

```python
# Preguntas de prueba
TEST_QUERIES = [
    "Tu pregunta 1",
    "Tu pregunta 2",
    # Añade más...
]

# Parámetros de recuperación
k_text = 1      # Número de documentos de texto a recuperar
k_img = 1       # Número de imágenes a recuperar
```

### Variables Modificables en `rag_metrics_integration.py`

```python
# Umbrales de calidad mínima
min_context_length = 100   # Caracteres mínimos esperados
min_response_length = 50   # Caracteres mínimos en respuesta
```

## 📊 Archivos Generados

- **ragas_results.json**: Resultados detallados de RAGAS
- **ragas_results.csv**: Mismos resultados en formato tabular
- **ragas_evaluations/latest_results.json**: Último resultado
- **ragas_evaluations/metrics_history.csv**: Histórico de evaluaciones
- **session_metrics.json**: Métricas de sesión actual
- **metrics_report_YYYYMMDD_HHMMSS.csv**: Reportes exportados

## 🎯 Mejora de Resultados

### Si FAITHFULNESS es baja (<0.6):
- ✓ Añade más documentos de referencia
- ✓ Mejora prompts para incluir "Solo usa contexto"
- ✓ Verifica que el retriever devuelva documentos relevantes

### Si ANSWER_RELEVANCY es baja (<0.6):
- ✓ Refina prompts para ser más directos
- ✓ Aumenta número de documentos recuperados
- ✓ Mejora embedding model

### Si CONTEXT_RELEVANCY es baja (<0.6):
- ✓ Mejora embedding model
- ✓ Aumenta número de resultados recuperados
- ✓ Ajusta threshold de similitud

### Si CONTEXT_PRECISION es baja (<0.6):
- ✓ Reduce número de documentos recuperados
- ✓ Mejora calidad de chunking
- ✓ Filtra documentos menos relevantes

## 💡 Mejores Prácticas

1. **Ejecuta regularmente**: Al menos una vez por semana
2. **Mantén histórico**: Monitorea tendencias de calidad
3. **Usa múltiples queries**: Variedad en tus preguntas de prueba
4. **Combina métricas**: No confíes en una sola métrica
5. **Valida manualmente**: Revisa respuestas que tengas dudas
6. **Documenta cambios**: Registra qué cambios afectaron métricas

## 🚨 Troubleshooting

### Error: "GOOGLE_API_KEY no encontrada"
```bash
# Crea .env si no existe
echo "GOOGLE_API_KEY=tu_api_key" > .env
```

### Error: "No module named ragas"
```bash
pip install ragas
```

### Error: Evaluación muy lenta
- Reduce TEST_QUERIES a 5 preguntas
- Usa modelo más ligero en `gemini_llm.py`
- Aumenta `k_text` y reduce `k_img`

### Resultados inconsistentes entre ejecuciones
- Normal para LLMs, la aleatoriedad es inherente
- Ejecuta varias veces y promedía resultados
- Usa métricas determinísticas como context_precision

## 📚 Recursos

- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [RAGAS Docs](https://docs.ragas.io/)
- [Paper RAGAS](https://arxiv.org/abs/2309.15217)

## 🔄 Flujo Recomendado

```
1. Ejecutar ragas_evaluator.py
   ↓
2. Revisar ragas_results.json
   ↓
3. Si scores < 0.7 → Hacer mejoras en código
   ↓
4. Ejecutar metrics_dashboard.py dashboard
   ↓
5. Comparar histórico con ejecución anterior
   ↓
6. Documentar cambios y resultados
   ↓
7. Repetir semanalmente
```

## ✅ Checklist de Calidad

- [ ] Faithfulness > 0.75
- [ ] Answer Relevancy > 0.70
- [ ] Context Relevancy > 0.75
- [ ] Context Precision > 0.75
- [ ] Latencia promedio < 5 segundos
- [ ] Tasa de errores < 5%
- [ ] Histórico muestra tendencia positiva

¡Si cumples estos criterios, tu RAG está en buen estado! 🎉
