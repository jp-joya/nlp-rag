from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_gemini import rag_answer
import re
import os
import csv
from pathlib import Path

app = Flask(__name__, static_folder='front', static_url_path='')
CORS(app)

def parse_references(context_text):
    """Extrae referencias del contexto de texto con su contenido."""
    references = []
    
    # Busca patrones como: [Fuente: nombre_archivo]
    pattern = r'\[Fuente:\s*([^\]]+)\]\s*\n(.*?)(?=(?:\n|$)(?:- \[Fuente:|$))'
    matches = re.findall(pattern, context_text, re.DOTALL)
    
    seen_sources = set()
    for source, content in matches:
        source = source.strip()
        if source not in seen_sources:  # evitar duplicados
            seen_sources.add(source)
            references.append({
                'source': source,
                'content': content.strip()[:300],  # limitar a 300 caracteres
                'type': 'texto'
            })
    
    return references

def parse_images(context_images):
    """Extrae rutas de imágenes del contexto de imágenes."""
    # Busca patrones como [Imagen: ruta/imagen.jpg]
    pattern = r'\[Imagen:\s*([^\]]+)\]'
    image_matches = re.findall(pattern, context_images)
    
    # Extraer solo el nombre del archivo (último componente de la ruta)
    import os
    image_names = [os.path.basename(img.strip()) for img in image_matches]
    
    # También busca descripciones
    description_pattern = r'Descripción:\s*([^\n]+)'
    descriptions = re.findall(description_pattern, context_images)
    
    return image_names, descriptions

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Query vacía'}), 400
        
        result = rag_answer(user_query)
        
        # Extraer referencias e imágenes
        references = parse_references(result['context_text'])
        image_paths, image_descriptions = parse_images(result['context_images'])
        
        # Construir lista de imágenes con sus descripciones
        images_with_desc = []
        for idx, path in enumerate(image_paths):
            description = image_descriptions[idx] if idx < len(image_descriptions) else 'Sin descripción'
            images_with_desc.append({
                'path': path.strip(),
                'description': description.strip()
            })
        
        return jsonify({
            'query': result['query'],
            'answer': result['answer'],
            'context_text': result['context_text'][:500],
            'context_images': result['context_images'][:500],
            'references': references,
            'images': images_with_desc
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<path:filename>', methods=['GET'])
def get_image(filename):
    """Sirve imágenes desde el directorio data/raw/images/"""
    from flask import send_file
    
    image_path = os.path.join('./data/raw/images', filename)
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Imagen no encontrada'}), 404
    
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    """Obtiene lista de experimentos disponibles"""
    try:
        experiments_dir = Path('./rag_eval/experiments')
        
        if not experiments_dir.exists():
            return jsonify({'error': 'Directorio de experimentos no encontrado'}), 404
        
        # Obtener todos los archivos CSV
        csv_files = list(experiments_dir.glob('*.csv'))
        
        experiments = []
        for csv_file in sorted(csv_files):
            # Leer el CSV para obtener estadísticas
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if rows:
                    total = len(rows)
                    passed = sum(1 for r in rows if r.get('score') == 'pass')
                    failed = total - passed
                    success_rate = (passed / total * 100) if total > 0 else 0
                    
                    experiments.append({
                        'name': csv_file.stem,
                        'filename': csv_file.name,
                        'total': total,
                        'passed': passed,
                        'failed': failed,
                        'success_rate': round(success_rate, 1)
                    })
        
        return jsonify({
            'experiments': experiments,
            'count': len(experiments)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiments/<experiment_name>', methods=['GET'])
def get_experiment_details(experiment_name):
    """Obtiene detalles completos de un experimento"""
    try:
        experiment_path = Path(f'./rag_eval/experiments/{experiment_name}.csv')
        
        if not experiment_path.exists():
            return jsonify({'error': 'Experimento no encontrado'}), 404
        
        # Leer el CSV
        with open(experiment_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return jsonify({'error': 'Experimento vacío'}), 400
        
        # Preparar datos
        total = len(rows)
        passed = sum(1 for r in rows if r.get('score') == 'pass')
        failed = total - passed
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Separar por resultado
        passed_tests = [r for r in rows if r.get('score') == 'pass']
        failed_tests = [r for r in rows if r.get('score') == 'fail']
        
        return jsonify({
            'name': experiment_name,
            'statistics': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'success_rate': round(success_rate, 1)
            },
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'all_tests': rows
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Sirve el archivo index.html del frontend"""
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def serve_frontend(path):
    """Sirve archivos estáticos del frontend (CSS, JS, etc.)"""
    return app.send_static_file(path)

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port=5000)
