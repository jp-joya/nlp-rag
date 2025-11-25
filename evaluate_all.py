"""
Main Entry Point - Script maestro para evaluación completa
Ejecuta todas las evaluaciones y genera reportes
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

print("\n" + "=" * 70)
print("🚀 RAG EVALUATION SUITE - MAIN CONTROLLER")
print("=" * 70 + "\n")


def run_ragas_evaluation():
    """Ejecuta evaluación RAGAS completa"""
    print("\n📊 INICIANDO EVALUACIÓN RAGAS...\n")
    try:
        from ragas_evaluator import main as ragas_main
        ragas_main()
        print("✅ Evaluación RAGAS completada")
        return True
    except Exception as e:
        print(f"❌ Error en evaluación RAGAS: {e}")
        return False


def run_metrics_dashboard():
    """Ejecuta dashboard de métricas"""
    print("\n📈 GENERANDO DASHBOARD DE MÉTRICAS...\n")
    try:
        from metrics_dashboard import display_metrics_dashboard, print_quick_status
        print_quick_status()
        display_metrics_dashboard()
        print("✅ Dashboard completado")
        return True
    except Exception as e:
        print(f"❌ Error en dashboard: {e}")
        return False


def run_quick_quality_check():
    """Ejecuta verificación rápida de calidad"""
    print("\n🔍 VERIFICACIÓN RÁPIDA DE CALIDAD...\n")
    try:
        from rag_metrics_integration import (
            measure_rag_performance,
            quick_quality_check,
            print_quality_report,
            save_session_metrics
        )
        from rag_gemini import rag_answer

        test_queries = [
            "¿Por qué es importante dormir bien?",
            "¿Cuáles son los macronutrientes?",
            "¿Cuánta agua debo beber?",
        ]

        total_score = 0
        for i, query in enumerate(test_queries, 1):
            print(f"\n  [{i}/{len(test_queries)}] Evaluando: {query[:50]}...")
            result = rag_answer(query)
            quality = quick_quality_check(query, result["context_text"], result["answer"])
            total_score += quality["overall_quality_score"]

        avg_score = total_score / len(test_queries)
        print(f"\n  📊 Puntuación promedio: {avg_score:.4f}")

        # Guardar sesión
        save_session_metrics("quick_quality_check.json")
        print("✅ Verificación rápida completada")
        return True

    except Exception as e:
        print(f"❌ Error en verificación: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_full_report():
    """Genera reporte completo en HTML"""
    print("\n📄 GENERANDO REPORTE COMPLETO...\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = f"rag_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Evaluation Report</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 40px;
            }}
            h1 {{
                color: #667eea;
                margin-bottom: 10px;
                text-align: center;
            }}
            .timestamp {{
                text-align: center;
                color: #999;
                margin-bottom: 30px;
                font-size: 14px;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                border-left: 4px solid #667eea;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            .section h2 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 18px;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }}
            .metric-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #764ba2;
            }}
            .metric-name {{
                font-weight: 600;
                color: #667eea;
                font-size: 14px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #764ba2;
                margin-top: 10px;
            }}
            .status-badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                margin-top: 10px;
            }}
            .status-good {{
                background: #d4edda;
                color: #155724;
            }}
            .status-warning {{
                background: #fff3cd;
                color: #856404;
            }}
            .status-bad {{
                background: #f8d7da;
                color: #721c24;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                text-align: center;
                font-size: 12px;
                color: #999;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }}
            th {{
                background: #667eea;
                color: white;
                font-weight: 600;
            }}
            tr:nth-child(even) {{
                background: #f9f9f9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 RAG Evaluation Report</h1>
            <div class="timestamp">Generado: {timestamp}</div>
            
            <div class="section">
                <h2>📋 Resumen Ejecutivo</h2>
                <p>
                    Este reporte contiene los resultados de evaluación del sistema RAG usando RAGAS 
                    (Retrieval-Augmented Generation Assessment Framework).
                </p>
            </div>

            <div class="section">
                <h2>🎯 Métricas Principales</h2>
                <p>Se han evaluado 4 dimensiones clave del sistema RAG:</p>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>Faithfulness</strong>: Fidelidad de la respuesta al contexto</li>
                    <li><strong>Answer Relevancy</strong>: Relevancia de la respuesta a la pregunta</li>
                    <li><strong>Context Relevancy</strong>: Relevancia del contexto recuperado</li>
                    <li><strong>Context Precision</strong>: Precisión del contexto recuperado</li>
                </ul>
            </div>

            <div class="section">
                <h2>📊 Resultados Detallados</h2>
                <p>Para ver resultados completos, consulta:</p>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><code>ragas_results.json</code> - Resultados en formato JSON</li>
                    <li><code>ragas_results.csv</code> - Resultados en formato CSV</li>
                    <li><code>ragas_evaluations/metrics_history.csv</code> - Histórico de evaluaciones</li>
                </ul>
            </div>

            <div class="section">
                <h2>💡 Recomendaciones</h2>
                <ul style="margin-left: 20px;">
                    <li>✓ Ejecuta evaluaciones regularmente (al menos semanal)</li>
                    <li>✓ Monitorea tendencias en métricas</li>
                    <li>✓ Usa dashboard para seguimiento visual</li>
                    <li>✓ Documenta cambios en código antes de evaluar</li>
                    <li>✓ Mejora iterativamente basado en resultados</li>
                </ul>
            </div>

            <div class="section">
                <h2>📈 Próximos Pasos</h2>
                <ol style="margin-left: 20px;">
                    <li>Revisar <code>RAGAS_GUIDE.md</code> para documentación completa</li>
                    <li>Ejecutar: <code>python metrics_dashboard.py dashboard</code></li>
                    <li>Analizar métricas y identificar áreas de mejora</li>
                    <li>Implementar mejoras en el código</li>
                    <li>Re-evaluar después de cambios</li>
                </ol>
            </div>

            <div class="footer">
                <p>RAG Evaluation Suite | Powered by RAGAS Framework</p>
                <p>📚 Documentación: RAGAS_GUIDE.md</p>
            </div>
        </div>
    </body>
    </html>
    """

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Reporte HTML generado: {report_path}")
    return True


def main():
    """Menú principal"""
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Suite - Evaluación completa de sistemas RAG"
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        choices=[
            "full",
            "ragas",
            "dashboard",
            "quick",
            "report",
            "help",
        ],
        help="Comando a ejecutar",
    )

    args = parser.parse_args()

    if args.command == "full":
        print("🔄 Ejecutando evaluación COMPLETA...\n")
        success = True
        success &= run_ragas_evaluation()
        success &= run_metrics_dashboard()
        success &= generate_full_report()

        if success:
            print("\n" + "=" * 70)
            print("✅ EVALUACIÓN COMPLETA EXITOSA")
            print("=" * 70)
            print("\n📁 Archivos generados:")
            print("  - ragas_results.json")
            print("  - ragas_results.csv")
            print("  - ragas_evaluation_report_*.html")
            print("  - ragas_evaluations/metrics_history.csv\n")
        else:
            print("\n" + "=" * 70)
            print("⚠️ EVALUACIÓN COMPLETADA CON ADVERTENCIAS")
            print("=" * 70 + "\n")

    elif args.command == "ragas":
        print("🔄 Ejecutando RAGAS...\n")
        run_ragas_evaluation()

    elif args.command == "dashboard":
        print("🔄 Mostrando DASHBOARD...\n")
        run_metrics_dashboard()

    elif args.command == "quick":
        print("🔄 Verificación RÁPIDA...\n")
        run_quick_quality_check()

    elif args.command == "report":
        print("🔄 Generando REPORTE...\n")
        generate_full_report()

    elif args.command == "help":
        print("""
COMANDOS DISPONIBLES:

  full       - Evaluación COMPLETA (RAGAS + Dashboard + Reporte)
  ragas      - Solo RAGAS evaluation
  dashboard  - Solo metrics dashboard
  quick      - Verificación rápida de calidad
  report     - Generar reporte HTML

EJEMPLOS:
  
  python evaluate_all.py full
  python evaluate_all.py ragas
  python evaluate_all.py dashboard
  python evaluate_all.py quick

DOCUMENTACIÓN:
  
  Ver RAGAS_GUIDE.md para guía completa
        """)

    print("\n" + "=" * 70)
    print("✨ Evaluación completada")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluación interrumpida por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
