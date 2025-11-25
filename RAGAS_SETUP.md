# 🚀 RAGAS Integration - Resumen

He integrado un framework completo de evaluación RAGAS para tu proyecto RAG. Aquí está lo que incluye:

## 📦 Componentes Creados

### 1. **ragas_evaluator.py** ⭐ Principal
Script que ejecuta evaluación RAGAS completa:
- Genera dataset de prueba automáticamente
- Evalúa 4 métricas RAGAS clave
- Genera reportes en JSON y CSV
- Muestra resumen visual

**Uso**:
```bash
python ragas_evaluator.py
```

**Output**:
- `ragas_results.json` - Resultados detallados
- `ragas_results.csv` - Resultados en tabla
- Resumen visual en terminal

---

### 2. **metrics_dashboard.py** 📊 Monitoreo
Dashboard para visualizar y monitorear métricas:
- Ver estado rápido de métricas
- Histórico de evaluaciones
- Exportar reportes (CSV/HTML)
- Detectar tendencias

**Comandos**:
```bash
# Estado rápido
python metrics_dashboard.py status

# Ver histórico completo
python metrics_dashboard.py dashboard

# Exportar reporte
python metrics_dashboard.py export csv
python metrics_dashboard.py export html
```

---

### 3. **rag_metrics_integration.py** 🔧 Integración
Integra métricas en tu código RAG:

**Características**:
- Decorador `@measure_rag_performance` para medir latencia
- `quick_quality_check()` para evaluación sin RAGAS
- `RAGMetricsCollector` para recolectar datos
- `print_quality_report()` para visualización

**Ejemplo de uso**:
```python
from rag_metrics_integration import (
    measure_rag_performance,
    quick_quality_check,
    print_quality_report
)

@measure_rag_performance
def mi_query(query):
    return rag_answer(query)

result = mi_query("¿Qué son macronutrientes?")
quality = quick_quality_check(result['query'], 
                             result['context_text'],
                             result['answer'])
print_quality_report(quality)
```

---

### 4. **evaluate_all.py** 🎯 Controlador Principal
Script maestro que ejecuta todo:

**Comandos principales**:
```bash
# Evaluación completa (todo en uno)
python evaluate_all.py full

# Solo RAGAS
python evaluate_all.py ragas

# Ver dashboard
python evaluate_all.py dashboard

# Verificación rápida
python evaluate_all.py quick

# Generar reporte HTML
python evaluate_all.py report

# Ver ayuda
python evaluate_all.py help
```

---

### 5. **RAGAS_GUIDE.md** 📚 Documentación
Guía completa sobre:
- ¿Qué es RAGAS?
- Explicación de cada métrica
- Cómo usar cada script
- Interpretación de resultados
- Troubleshooting
- Mejores prácticas

---

## 🎯 Métricas Evaluadas

| Métrica | ¿Qué mide? | Interpretación |
|---------|-----------|-----------------|
| **Faithfulness** | ¿Respuesta basada en contexto? | 0.8+ = Excelente |
| **Answer Relevancy** | ¿Respuesta relevante? | 0.8+ = Muy relevante |
| **Context Relevancy** | ¿Contexto relevante? | 0.8+ = Contexto excelente |
| **Context Precision** | ¿Contexto preciso? | 0.8+ = Muy preciso |

---

## 📊 Flujo de Trabajo

```
1. Ejecuta evaluación completa
   └─ python evaluate_all.py full

2. Se generan métricas
   ├─ ragas_results.json
   ├─ ragas_results.csv
   └─ ragas_evaluation_report_*.html

3. Monitorea histórico
   └─ python metrics_dashboard.py dashboard

4. Identifica áreas de mejora
   └─ Analiza qué métricas son bajas

5. Implementa cambios
   └─ Mejora código/prompts/retriever

6. Re-evalúa
   └─ Repite paso 1
```

---

## 🚀 Quick Start

### Primer uso (Recomendado):
```bash
# 1. Ejecuta evaluación completa
python evaluate_all.py full

# Esto genera:
# - Evaluación RAGAS
# - Dashboard de métricas
# - Reporte HTML
```

### Uso regular:
```bash
# 2. Ver estado rápido
python metrics_dashboard.py status

# 3. Ver histórico
python metrics_dashboard.py dashboard

# 4. Re-evaluar después de cambios
python evaluate_all.py ragas
```

---

## 📁 Archivos Generados

```
ragas_results.json              # Resultados RAGAS (JSON)
ragas_results.csv               # Resultados RAGAS (CSV)
ragas_evaluation_report_*.html  # Reporte HTML
ragas_evaluations/
├── latest_results.json         # Último resultado
├── metrics_history.csv         # Histórico de métricas
session_metrics.json            # Métricas de sesión actual
quick_quality_check.json        # Resultado de verificación rápida
```

---

## 💡 Casos de Uso

### Caso 1: Verificación Rápida (1-2 minutos)
```bash
python evaluate_all.py quick
```
✓ No necesita RAGAS
✓ Rápido
✓ Bueno para desarrollo

### Caso 2: Evaluación Completa (5-15 minutos)
```bash
python evaluate_all.py full
```
✓ Evaluación RAGAS completa
✓ Histórico de métricas
✓ Reporte HTML

### Caso 3: Monitoreo (segundos)
```bash
python metrics_dashboard.py status
```
✓ Ver estado actual
✓ Muy rápido
✓ Ideal para chequeos frecuentes

---

## 🔧 Configuración

### Personalizar preguntas de prueba
Edita `ragas_evaluator.py`:
```python
TEST_QUERIES = [
    "Tu pregunta 1",
    "Tu pregunta 2",
    # Añade más...
]
```

### Ajustar umbrales de calidad
Edita `rag_metrics_integration.py`:
```python
RAGQualityEvaluator.evaluate_context_length(context, min_length=100)
RAGQualityEvaluator.evaluate_response_length(response, min_length=50)
```

---

## ✅ Objetivos de Calidad

Para un RAG en buen estado:
- ✓ Faithfulness > 0.75
- ✓ Answer Relevancy > 0.70
- ✓ Context Relevancy > 0.75
- ✓ Context Precision > 0.75
- ✓ Latencia promedio < 5s
- ✓ Tasa de errores < 5%

---

## 📖 Documentación Completa

Para más detalles, consulta: **RAGAS_GUIDE.md**

---

## 🆘 Troubleshooting

**Error: "No module named ragas"**
```bash
pip install ragas
```

**Error: "GOOGLE_API_KEY no encontrada"**
```bash
echo "GOOGLE_API_KEY=tu_api_key" > .env
```

**Evaluación muy lenta**
- Usa `python evaluate_all.py quick` para desarrollo
- Reduce TEST_QUERIES a 3-5 preguntas
- Usa RAGAS solo semanalmente

---

## 🎓 Próximos Pasos

1. ✅ Lee **RAGAS_GUIDE.md**
2. ✅ Ejecuta `python evaluate_all.py full`
3. ✅ Analiza resultados
4. ✅ Identifica áreas de mejora
5. ✅ Implementa cambios
6. ✅ Re-evalúa regularmente

---

**¡Tu RAG ahora tiene visibilidad completa en su rendimiento! 📊✨**
