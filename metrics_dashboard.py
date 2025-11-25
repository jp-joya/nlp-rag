"""
Metrics Dashboard - Monitoreo continuo de RAG
Ejecuta evaluaciones periódicas y mantiene histórico
"""

import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

RESULTS_DIR = "./ragas_evaluations"
METRICS_LOG = os.path.join(RESULTS_DIR, "metrics_history.csv")
LATEST_RESULTS = os.path.join(RESULTS_DIR, "latest_results.json")


def ensure_results_dir():
    """Crea directorio para resultados si no existe"""
    Path(RESULTS_DIR).mkdir(exist_ok=True)


def load_latest_results() -> dict:
    """Carga resultados más recientes"""
    if os.path.exists(LATEST_RESULTS):
        with open(LATEST_RESULTS, "r") as f:
            return json.load(f)
    return None


def save_metrics_snapshot(metrics_summary: dict):
    """Guarda snapshot actual en histórico"""
    ensure_results_dir()

    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # Guardar último resultado
    with open(LATEST_RESULTS, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Agregar a histórico CSV
    metrics = metrics_summary.get("metrics", {})
    history_row = {
        "timestamp": timestamp_str,
        "total_questions": metrics_summary.get("total_questions", 0),
    }

    # Agregar cada métrica
    for metric_name, values in metrics.items():
        if isinstance(values, dict):
            history_row[f"{metric_name}_mean"] = values.get("mean")
            history_row[f"{metric_name}_min"] = values.get("min")
            history_row[f"{metric_name}_max"] = values.get("max")

    # Añadir a CSV
    if os.path.exists(METRICS_LOG):
        df_existing = pd.read_csv(METRICS_LOG)
        df_new = pd.DataFrame([history_row])
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = pd.DataFrame([history_row])

    df.to_csv(METRICS_LOG, index=False)

    print(f"📊 Métricas guardadas en histórico: {METRICS_LOG}")


def display_metrics_dashboard():
    """Muestra dashboard de métricas del histórico"""
    if not os.path.exists(METRICS_LOG):
        print("❌ No hay histórico de métricas. Ejecuta ragas_evaluator.py primero.")
        return

    df = pd.read_csv(METRICS_LOG)

    print("\n" + "=" * 80)
    print("📊 METRICS DASHBOARD - HISTORICAL OVERVIEW")
    print("=" * 80)

    print(f"\n📈 Total evaluaciones: {len(df)}")
    print(f"📅 Período: {df['timestamp'].iloc[0]} a {df['timestamp'].iloc[-1]}")

    print("\n" + "-" * 80)
    print("RESUMEN POR MÉTRICA:")
    print("-" * 80)

    # Identificar métricas disponibles
    metric_cols = [col for col in df.columns if col.endswith("_mean")]

    for metric_col in metric_cols:
        metric_name = metric_col.replace("_mean", "").upper()
        print(f"\n{metric_name}:")

        values = df[metric_col].dropna()
        if len(values) > 0:
            print(f"  Promedio histórico: {values.mean():.4f}")
            print(f"  Tendencia:         {values.iloc[-1]:.4f} (última evaluación)")
            print(f"  Rango:            {values.min():.4f} - {values.max():.4f}")

            # Detectar tendencia
            if len(values) > 1:
                recent_mean = values.iloc[-5:].mean() if len(values) >= 5 else values.mean()
                old_mean = values.iloc[:5].mean() if len(values) >= 5 else values.mean()
                if recent_mean > old_mean:
                    print(f"  📈 Tendencia: MEJORA")
                elif recent_mean < old_mean:
                    print(f"  📉 Tendencia: EMPEORA")
                else:
                    print(f"  ➡️ Tendencia: ESTABLE")

    print("\n" + "=" * 80)
    print("ÚLTIMOS 5 REGISTROS:")
    print("-" * 80)
    print(df.tail(5).to_string(index=False))

    print("\n" + "=" * 80 + "\n")


def export_metrics_report(format: str = "csv"):
    """Exporta reporte de métricas"""
    if not os.path.exists(METRICS_LOG):
        print("❌ No hay métricas para exportar")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "csv":
        export_path = f"./metrics_report_{timestamp}.csv"
        df = pd.read_csv(METRICS_LOG)
        df.to_csv(export_path, index=False)
        print(f"✅ Reporte exportado: {export_path}")

    elif format == "html":
        export_path = f"./metrics_report_{timestamp}.html"
        df = pd.read_csv(METRICS_LOG)
        
        html = df.to_html(index=False)
        with open(export_path, "w") as f:
            f.write(f"""
            <html>
                <head>
                    <title>RAG Metrics Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #667eea; color: white; }}
                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <h1>RAG Evaluation Metrics Report</h1>
                    <p>Generated: {datetime.now()}</p>
                    {html}
                </body>
            </html>
            """)
        print(f"✅ Reporte HTML exportado: {export_path}")


def get_metrics_status():
    """Retorna estado actual de las métricas"""
    latest = load_latest_results()

    if not latest:
        return {
            "status": "NO_DATA",
            "message": "Sin datos de evaluación",
        }

    metrics = latest.get("metrics", {})
    status = {
        "timestamp": latest.get("timestamp"),
        "total_questions": latest.get("total_questions"),
        "metrics": {},
    }

    for metric_name, values in metrics.items():
        if isinstance(values, dict):
            mean = values.get("mean", 0)
            if mean >= 0.8:
                health = "🟢 EXCELENTE"
            elif mean >= 0.6:
                health = "🟡 BUENO"
            else:
                health = "🔴 NECESITA MEJORA"

            status["metrics"][metric_name] = {
                "score": mean,
                "health": health,
            }

    return status


def print_quick_status():
    """Imprime estado rápido de las métricas"""
    status = get_metrics_status()

    print("\n" + "=" * 60)
    print("⚡ QUICK METRICS STATUS")
    print("=" * 60)

    if status["status"] == "NO_DATA":
        print("❌ Sin datos de evaluación")
        print("   Ejecuta: python ragas_evaluator.py")
    else:
        print(f"📅 Última actualización: {status['timestamp']}")
        print(f"❓ Preguntas evaluadas: {status['total_questions']}\n")

        for metric_name, metric_data in status["metrics"].items():
            score = metric_data["score"]
            health = metric_data["health"]
            print(f"{health} {metric_name}: {score:.4f}")

    print("=" * 60 + "\n")


def main():
    """Menú principal del dashboard"""
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "dashboard":
            display_metrics_dashboard()
        elif cmd == "status":
            print_quick_status()
        elif cmd == "export":
            format_type = sys.argv[2] if len(sys.argv) > 2 else "csv"
            export_metrics_report(format_type)
        else:
            print("Comandos disponibles:")
            print("  python metrics_dashboard.py dashboard  - Ver histórico completo")
            print("  python metrics_dashboard.py status     - Estado rápido")
            print("  python metrics_dashboard.py export csv/html - Exportar reporte")
    else:
        print_quick_status()
        display_metrics_dashboard()


if __name__ == "__main__":
    main()
