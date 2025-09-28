# webapp/app.py - Flask web interface for the RAG system

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import json
from pathlib import Path
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MultimodalRAG
from indexer.faiss_index import get_index, get_index_stats
from rag.llm_client import is_model_available

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize RAG system
rag_system = MultimodalRAG()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Main page"""
    stats = rag_system.get_stats()
    return render_template('index.html', stats=stats)

@app.route('/query', methods=['POST'])
def query():
    """Handle queries"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        # Get number of results
        k = data.get('k', 5)
        k = min(max(k, 1), 20)  # Limit between 1 and 20

        # Query the system
        response = rag_system.query(question, k=k)

        # Get search results for display
        from embeddings.embedder import text_to_embedding
        query_embedding = text_to_embedding(question)
        results = get_index().search(query_embedding, k=k)

        # Format results for display
        formatted_results = []
        for result in results:
            doc = result.get('document', {})
            formatted_results.append({
                'file_path': doc.get('file_path', 'Unknown'),
                'file_type': doc.get('file_type', 'Unknown'),
                'snippet': result.get('snippet', ''),
                'score': result.get('score', 0),
                'metadata': doc.get('metadata', {})
            })

        return jsonify({
            'response': response,
            'results': formatted_results,
            'total_results': len(formatted_results)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '' or file.filename is None:
            return jsonify({'error': 'No file selected'}), 400

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Ingest file
        success = rag_system.ingest_file(file_path)

        if success:
            # Save index
            from indexer.faiss_index import save_index
            save_index()

            return jsonify({
                'message': f'File {filename} uploaded and ingested successfully',
                'file_path': file_path
            })
        else:
            # Clean up failed upload
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'Failed to ingest file'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ingest_directory', methods=['POST'])
def ingest_directory():
    """Ingest a directory"""
    try:
        data = request.get_json()
        directory_path = data.get('path', '').strip()

        if not directory_path:
            return jsonify({'error': 'Directory path is required'}), 400

        if not os.path.exists(directory_path):
            return jsonify({'error': 'Directory does not exist'}), 404

        # Ingest directory
        success = rag_system.ingest_directory(directory_path)

        if success:
            # Save index
            from indexer.faiss_index import save_index
            save_index()

            return jsonify({'message': f'Directory {directory_path} ingested successfully'})
        else:
            return jsonify({'error': 'Failed to ingest directory'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get system statistics"""
    try:
        stats = rag_system.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/files/<path:filename>')
def serve_file(filename):
    """Serve uploaded files"""
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except Exception as e:
        return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_available': is_model_available(),
        'index_stats': get_index_stats()
    })

if __name__ == '__main__':
    print("Starting Multimodal RAG Web Interface...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='localhost', port=5000)
