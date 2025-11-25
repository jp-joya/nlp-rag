from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_gemini import rag_answer

app = Flask(__name__)
CORS(app)

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Query vacía'}), 400
        
        result = rag_answer(user_query)

        print(result)
        
        return jsonify({
            'query': result['query'],
            'answer': result['answer'],
            'context_text': result['context_text'][:300],  # truncar para UI
            'context_images': result['context_images'][:300]
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)