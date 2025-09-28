#!/usr/bin/env python3
# main.py - Main RAG application orchestrator

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.install_check import run_system_checks
from embeddings.embedder import get_embedding_dim
from indexer.faiss_index import get_index, get_index_stats, save_index
from rag.prompt_builder import build_prompt
from rag.llm_client import get_llm_client, is_model_available
from ingesters import ingest_docs, ingest_images, ingest_audio

class MultimodalRAG:
    """Main RAG application class"""

    def __init__(self, data_dir="data", db_dir="db", models_dir="models"):
        """Initialize the RAG system"""
        self.data_dir = Path(data_dir)
        self.db_dir = Path(db_dir)
        self.models_dir = Path(models_dir)

        # Create directories
        for dir_path in [self.data_dir, self.db_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)

        print("Multimodal RAG System Initialized")
        print(f"Data directory: {self.data_dir}")
        print(f"Database directory: {self.db_dir}")
        print(f"Models directory: {self.models_dir}")

    def check_dependencies(self):
        """Check and install dependencies"""
        print("Checking dependencies...")
        return run_system_checks()

    def ingest_directory(self, directory_path):
        """Ingest all files in a directory"""
        dir_path = Path(directory_path)
        if not dir_path.exists():
            print(f"Directory not found: {directory_path}")
            return False

        success_count = 0
        total_count = 0

        # Process all files
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                total_count += 1
                if self.ingest_file(str(file_path)):
                    success_count += 1

        print(f"Ingestion complete: {success_count}/{total_count} files processed")
        return success_count > 0

    def ingest_file(self, file_path):
        """Ingest a single file"""
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return False

        # Determine file type and ingest accordingly
        suffix = file_path.suffix.lower()

        if suffix in ['.pdf', '.docx', '.doc']:
            return ingest_docs.ingest_document(str(file_path))
        elif suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            return ingest_images.ingest_image(str(file_path))
        elif suffix in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']:
            return ingest_audio.ingest_audio(str(file_path))
        else:
            print(f"Unsupported file type: {suffix}")
            return False

    def query(self, question, k=5):
        """Query the RAG system"""
        if not is_model_available():
            print("LLM model not available. Please set up a model first.")
            return None

        try:
            # Get query embedding
            from embeddings.embedder import text_to_embedding
            query_embedding = text_to_embedding(question)

            # Search index
            results = get_index().search(query_embedding, k=k)

            if not results:
                print("No relevant information found.")
                return "I don't know - no relevant information found in the available sources."

            # Build prompt
            prompt = build_prompt(question, results)

            # Generate response
            llm_client = get_llm_client()
            response = llm_client.generate(prompt, max_tokens=512, temperature=0.1)

            return response

        except Exception as e:
            print(f"Error during query: {e}")
            return f"Error processing query: {e}"

    def get_stats(self):
        """Get system statistics"""
        index_stats = get_index_stats()
        model_available = is_model_available()

        return {
            "index": index_stats,
            "model_available": model_available,
            "data_directory": str(self.data_dir),
            "db_directory": str(self.db_dir),
            "models_directory": str(self.models_dir)
        }

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Multimodal RAG System")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--db-dir", default="db", help="Database directory")
    parser.add_argument("--models-dir", default="models", help="Models directory")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    subparsers.add_parser("setup", help="Check and install dependencies")

    # Ingest commands
    ingest_parser = subparsers.add_parser("ingest", help="Ingest files")
    ingest_parser.add_argument("path", help="File or directory to ingest")
    ingest_parser.add_argument("--file", action="store_true", help="Treat path as single file")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")

    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")

    # Interactive command
    subparsers.add_parser("interactive", help="Interactive mode")

    args = parser.parse_args()

    # Initialize RAG system
    rag = MultimodalRAG(args.data_dir, args.db_dir, args.models_dir)

    # Process commands
    if args.command == "setup":
        success = rag.check_dependencies()
        if success:
            print("Setup complete!")
        else:
            print("Setup failed. Please install missing dependencies.")
            sys.exit(1)

    elif args.command == "ingest":
        if args.file:
            success = rag.ingest_file(args.path)
        else:
            success = rag.ingest_directory(args.path)

        if success:
            save_index()  # Save the updated index
        else:
            print("Ingestion failed.")

    elif args.command == "query":
        response = rag.query(args.question, k=args.k)
        if response:
            print("\n" + "="*50)
            print("RESPONSE:")
            print("="*50)
            print(response)
            print("="*50)

    elif args.command == "stats":
        stats = rag.get_stats()
        print("\n" + "="*50)
        print("SYSTEM STATISTICS:")
        print("="*50)
        print(f"Index: {stats['index']}")
        print(f"Model Available: {stats['model_available']}")
        print(f"Data Directory: {stats['data_directory']}")
        print(f"Database Directory: {stats['db_directory']}")
        print(f"Models Directory: {stats['models_directory']}")

    elif args.command == "interactive":
        print("Interactive Mode")
        print("Type 'quit' or 'exit' to exit")
        print("Type 'ingest <path>' to ingest files")
        print("Type 'stats' for system statistics")
        print()

        while True:
            try:
                user_input = input("RAG> ").strip()

                if user_input.lower() in ['quit', 'exit']:
                    break

                elif user_input.lower() == 'stats':
                    stats = rag.get_stats()
                    print(f"Index vectors: {stats['index']['total_vectors']}")
                    print(f"Model loaded: {stats['model_available']}")

                elif user_input.lower().startswith('ingest '):
                    path = user_input[7:].strip()
                    if path:
                        rag.ingest_directory(path)
                        save_index()
                    else:
                        print("Please provide a path to ingest")

                elif user_input:
                    response = rag.query(user_input)
                    if response:
                        print(f"Answer: {response}")

            except KeyboardInterrupt:
                print("\nUse 'quit' or 'exit' to exit")
            except Exception as e:
                print(f"Error: {e}")

    else:
        parser.print_help()

    # Save index on exit
    try:
        save_index()
    except:
        pass

if __name__ == "__main__":
    main()
