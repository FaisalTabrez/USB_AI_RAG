# indexer/faiss_index.py
import faiss
import numpy as np
import json
import os
import sqlite3
from embeddings.embedder import get_embedding_dim
import threading

class MetadataDB:
    """SQLite database for storing metadata and snippets"""

    def __init__(self, db_path="db/metadata.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        """Initialize the metadata database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content_type TEXT NOT NULL, -- 'text', 'image', 'audio'
                    metadata TEXT, -- JSON string
                    snippet TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_type ON documents(file_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_type ON documents(content_type)")

    def add_document(self, file_path, file_type, content_type, metadata, snippet):
        """Add a document to the metadata database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO documents (file_path, file_type, content_type, metadata, snippet)
                VALUES (?, ?, ?, ?, ?)
            """, (file_path, file_type, content_type, json.dumps(metadata), snippet))
            return cursor.lastrowid

    def get_document(self, doc_id):
        """Retrieve a document by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM documents WHERE id = ?
            """, (doc_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'file_path': row[1],
                    'file_type': row[2],
                    'content_type': row[3],
                    'metadata': json.loads(row[4]) if row[4] else {},
                    'snippet': row[5],
                    'created_at': row[6]
                }
            return None

    def search_documents(self, query, limit=10):
        """Search documents by content"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM documents
                WHERE snippet LIKE ?
                ORDER BY id DESC
                LIMIT ?
            """, (f'%{query}%', limit))
            return [
                {
                    'id': row[0],
                    'file_path': row[1],
                    'file_type': row[2],
                    'content_type': row[3],
                    'metadata': json.loads(row[4]) if row[4] else {},
                    'snippet': row[5],
                    'created_at': row[6]
                }
                for row in cursor.fetchall()
            ]

class FaissIndex:
    """FAISS-based vector index with metadata storage"""

    def __init__(self, index_file="db/faiss.index", metadata_db_path="db/metadata.db"):
        """Initialize FAISS index"""
        self.index_file = index_file
        self.metadata_db = MetadataDB(metadata_db_path)
        self.embedding_dim = get_embedding_dim()
        self.index = None
        self.id_to_doc_id = {}  # Map FAISS IDs to document IDs
        self.next_faiss_id = 0
        self.lock = threading.Lock()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_file), exist_ok=True)

        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
                # Load ID mapping if it exists
                mapping_file = self.index_file + ".mapping.json"
                if os.path.exists(mapping_file):
                    with open(mapping_file, 'r') as f:
                        self.id_to_doc_id = json.load(f)
                        self.next_faiss_id = max(self.id_to_doc_id.keys()) + 1 if self.id_to_doc_id else 0
                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading index: {e}")
                print("Creating new index...")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        # Use HNSW for fast approximate search
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        # Set search parameters for good quality/speed tradeoff
        faiss.ParameterSpace().set_index_parameter(self.index, "efSearch", 64)
        print(f"Created new FAISS index with dimension {self.embedding_dim}")

    def add_vector(self, vector, file_path, file_type, content_type, metadata, snippet):
        """Add a vector to the index"""
        with self.lock:
            if self.index is None:
                raise RuntimeError("Index not initialized")

            # Add to metadata database
            doc_id = self.metadata_db.add_document(file_path, file_type, content_type, metadata, snippet)

            # Add to FAISS index
            if vector.ndim == 1:
                vector = np.expand_dims(vector, 0)

            self.index.add(vector.astype('float32'))

            # Store mapping
            faiss_id = self.next_faiss_id
            self.id_to_doc_id[faiss_id] = doc_id
            self.next_faiss_id += 1

            # Save mapping to disk
            self._save_mapping()

            return faiss_id

    def _save_mapping(self):
        """Save ID mapping to disk"""
        mapping_file = self.index_file + ".mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(self.id_to_doc_id, f)

        # Save FAISS index
        faiss.write_index(self.index, self.index_file)

    def search(self, query_vector, k=8):
        """Search for similar vectors"""
        if self.index is None or self.index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, 0)

        # Search
        distances, indices = self.index.search(query_vector.astype('float32'), min(k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No more results
                continue

            # Get document from metadata database
            doc_id = self.id_to_doc_id.get(idx)
            if doc_id:
                doc = self.metadata_db.get_document(doc_id)
                if doc:
                    results.append({
                        'score': float(dist),
                        'distance': float(dist),
                        'faiss_id': int(idx),
                        'document': doc
                    })

        return results

    def get_stats(self):
        """Get index statistics"""
        total_vectors = self.index.ntotal if self.index else 0
        return {
            'total_vectors': total_vectors,
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.index).__name__ if self.index else 'None'
        }

    def save(self):
        """Save index and metadata"""
        with self.lock:
            if self.index:
                self._save_mapping()
                print(f"Saved index with {self.index.ntotal} vectors")
            else:
                print("No index to save")

    def close(self):
        """Close the index"""
        self.save()

# Global index instance
_index = None

def get_index():
    """Get or create the global index instance"""
    global _index
    if _index is None:
        _index = FaissIndex()
    return _index

def add_vector(vector, file_path, file_type, content_type, metadata, snippet):
    """Convenience function to add a vector to the index"""
    return get_index().add_vector(vector, file_path, file_type, content_type, metadata, snippet)

def search(query_vector, k=8):
    """Convenience function to search the index"""
    return get_index().search(query_vector, k)

def save_index():
    """Convenience function to save the index"""
    get_index().save()

def get_index_stats():
    """Convenience function to get index statistics"""
    return get_index().get_stats()
