# ingesters/ingest_docs.py
import os
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None
from embeddings.embedder import text_to_embedding
from indexer.faiss_index import add_vector

CHUNK_SIZE = 800  # characters
OVERLAP = 150  # characters

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks"""
    if not text or len(text) <= size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + size

        # Try to end at a sentence or word boundary
        if end < len(text):
            # Look for sentence endings
            sentence_end = max(
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end)
            )

            if sentence_end > start + size // 2:  # If we found a good break point
                end = sentence_end + 1
            else:
                # Look for word boundaries
                word_end = text.rfind(' ', start, end)
                if word_end > start + size // 2:
                    end = word_end

        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Move start position with overlap
        start = max(start + 1, end - overlap)

        # Prevent infinite loop
        if start >= end:
            break

    return chunks

def ingest_pdf(file_path):
    """Ingest a PDF file"""
    if fitz is None:
        print("PyMuPDF not available. Please install manually: pip install PyMuPDF")
        return False

    try:
        doc = fitz.open(file_path)
        print(f"Processing PDF: {file_path} ({len(doc)} pages)")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()

            if not text:
                continue

            # Chunk the page text
            chunks = chunk_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                # Get embedding
                embedding = text_to_embedding(chunk)

                # Prepare metadata
                metadata = {
                    "page": page_num + 1,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "page_text_length": len(text)
                }

                # Add to index
                add_vector(
                    vector=embedding[0],  # Take first (and only) embedding
                    file_path=file_path,
                    file_type="pdf",
                    content_type="text",
                    metadata=metadata,
                    snippet=chunk[:200] + "..." if len(chunk) > 200 else chunk
                )

        doc.close()
        print(f"Successfully ingested PDF: {file_path}")
        return True

    except Exception as e:
        print(f"Error ingesting PDF {file_path}: {e}")
        return False

def ingest_docx(file_path):
    """Ingest a DOCX file"""
    if docx is None:
        print("python-docx not available. Please install manually: pip install python-docx")
        return False

    try:
        doc = docx.Document(file_path)
        print(f"Processing DOCX: {file_path}")

        # Extract all text from paragraphs
        paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs]
        full_text = "\n".join([p for p in paragraphs if p])

        if not full_text:
            print(f"No text content found in {file_path}")
            return False

        # Chunk the full text
        chunks = chunk_text(full_text)

        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            # Get embedding
            embedding = text_to_embedding(chunk)

            # Prepare metadata
            metadata = {
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "document_text_length": len(full_text)
            }

            # Add to index
            add_vector(
                vector=embedding[0],
                file_path=file_path,
                file_type="docx",
                content_type="text",
                metadata=metadata,
                snippet=chunk[:200] + "..." if len(chunk) > 200 else chunk
            )

        print(f"Successfully ingested DOCX: {file_path}")
        return True

    except Exception as e:
        print(f"Error ingesting DOCX {file_path}: {e}")
        return False

def ingest_document(file_path):
    """Ingest a document file (PDF or DOCX)"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        return ingest_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        return ingest_docx(file_path)
    else:
        print(f"Unsupported document format: {file_ext}")
        return False

def batch_ingest_documents(directory_path):
    """Ingest all documents in a directory"""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return False

    success_count = 0
    total_count = 0

    # Supported extensions
    supported_exts = ['.pdf', '.docx', '.doc']

    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in supported_exts):
            file_path = os.path.join(directory_path, filename)
            total_count += 1

            if ingest_document(file_path):
                success_count += 1

    print(f"Batch ingestion complete: {success_count}/{total_count} documents processed")
    return success_count == total_count

if __name__ == "__main__":
    # Test the ingester
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.isdir(file_path):
            batch_ingest_documents(file_path)
        else:
            ingest_document(file_path)
    else:
        print("Usage: python ingest_docs.py <file_path_or_directory>")
