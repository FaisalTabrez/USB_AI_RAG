# Multimodal RAG System 🤖

A comprehensive Retrieval-Augmented Generation (RAG) system that can ingest, index, and query diverse data formats including PDF documents, DOCX files, images, screenshots, and audio recordings within a unified semantic retrieval framework.

## 🌟 Features

### Multimodal Support
- **Documents**: PDF and DOCX files with text extraction and chunking
- **Images**: OCR text extraction and visual embeddings
- **Audio**: Speech-to-text transcription with time-aligned chunks
- **Screenshots**: Enhanced image processing with metadata

### Advanced RAG Capabilities
- **Unified Vector Space**: All modalities indexed in shared semantic space
- **Cross-Modal Search**: Query across different content types seamlessly
- **Citation Transparency**: Every answer includes numbered citations with source locations
- **Offline Operation**: Complete USB/SSD-based operation with local LLM

### User Interface
- **Web Interface**: Modern Flask-based UI with real-time interactions
- **CLI Interface**: Command-line tools for batch operations
- **REST API**: Full API for integration with other systems

## 🏗️ Architecture

```
usbai/
├─ models/                 # LLM & embedding models (GGUF/ONNX/HF)
├─ db/                     # FAISS index + metadata store (json/sqlite)
├─ ingesters/
│   ├─ ingest_docs.py      # PDF/DOCX processing
│   ├─ ingest_images.py    # Image OCR + embeddings
│   └─ ingest_audio.py     # Audio transcription + chunking
├─ embeddings/
│   └─ embedder.py         # Multi-modal embedding pipeline
├─ indexer/
│   └─ faiss_index.py      # Vector indexing with metadata
├─ rag/
│   ├─ prompt_builder.py   # RAG prompt construction
│   └─ llm_client.py       # Local LLM integration
├─ webapp/                 # Flask UI
├─ utils/
│   └─ install_check.py    # Dependency management
└─ main.py                 # CLI orchestrator
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and navigate to the project
git clone <repository-url>
cd multimodal-rag-system

# Install Python dependencies
pip install -r requirements.txt

# Check system dependencies
python utils/install_check.py
```

### 2. Setup Dependencies

The system will automatically prompt you to install missing dependencies:

```bash
python main.py setup
```

**Required System Packages:**
- `tesseract` - OCR for image text extraction
- `ffmpeg` - Audio processing
- `whisper.cpp` (optional) - Fast offline speech recognition

**Required Python Packages:**
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Text embeddings
- `llama-cpp-python` - Local LLM inference
- `pytesseract` - OCR wrapper
- `PyMuPDF` - PDF processing
- `python-docx` - DOCX processing
- `pydub` - Audio manipulation

### 3. Download Models

You'll need to download a local LLM model. Recommended options:

**Small & Fast:**
- [Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)
- [Mistral-7B-Instruct-GGML](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGML)

**Place the model file in the `models/` directory:**
```bash
mkdir -p models
# Download your chosen model to models/
```

### 4. Ingest Your Data

**Via CLI:**
```bash
# Ingest single file
python main.py ingest path/to/document.pdf

# Ingest entire directory
python main.py ingest /path/to/documents/

# Ingest with file flag
python main.py ingest --file path/to/image.png
```

**Via Web Interface:**
1. Start the web server: `python webapp/app.py`
2. Open http://localhost:5000
3. Upload files or specify directory paths

### 5. Query Your Data

**Via CLI:**
```bash
python main.py query "What are the key findings in the annual report?"
python main.py query "Show me the screenshot taken at 14:32"
python main.py query "What was discussed in the meeting recording?"
```

**Via Web Interface:**
1. Open the web UI
2. Type your question in natural language
3. View results with citations and source locations

**Via Interactive Mode:**
```bash
python main.py interactive
```

## 💻 Usage Examples

### Text Queries
```
User: "What are the main conclusions from the Q4 report?"
Assistant: "The Q4 report shows a 15% increase in revenue compared to Q3, with the main growth drivers being the expansion into European markets and increased digital service adoption. [1]

Key findings include:
- Revenue growth: 15% QoQ
- Market expansion: 3 new European countries
- Digital adoption: 40% increase in online customers

SOURCES:
[1] /data/reports/Q4-2024.pdf (page 12)"
```

### Cross-Modal Queries
```
User: "Show me the report that references the screenshot taken at 14:32"
Assistant: "The screenshot taken at 14:32 showing the dashboard with real-time metrics is referenced in the weekly standup meeting notes from March 15th. The screenshot captures the moment when the team lead presented the new KPI tracking system. [1][2]

SOURCES:
[1] /data/screenshots/dashboard_2024-03-15_1432.png
[2] /data/meetings/standup_2024-03-15.txt (00:15:30-00:16:45)"
```

### Audio Queries
```
User: "What was discussed about project deadlines in the last team meeting?"
Assistant: "In the team meeting on March 10th, the project manager discussed extending the Q2 deadline by two weeks due to resource constraints in the development team. Sarah mentioned that the frontend integration was behind schedule, and Mike suggested bringing in external contractors to help meet the original timeline. [1]

SOURCES:
[1] /data/recordings/team_meeting_2024-03-10.wav (00:05:20-00:08:15)"
```

## 🔧 Configuration

### Directory Structure
```
data/           # Your documents, images, audio files
db/            # FAISS index and metadata (auto-created)
models/        # Local LLM models
uploads/       # Temporary file uploads (auto-created)
```

### Environment Variables
```bash
export RAG_DATA_DIR="/path/to/your/data"
export RAG_DB_DIR="/path/to/database"
export RAG_MODEL_PATH="/path/to/your/model.gguf"
```

### CLI Options
```bash
# Custom directories
python main.py --data-dir /my/data --db-dir /my/db --models-dir /my/models setup

# Query with custom parameters
python main.py query "your question" --k 10

# Interactive mode
python main.py interactive
```

## 🛠️ Advanced Features

### Custom Chunking
The system intelligently chunks content:
- **Documents**: 800-character chunks with 150-character overlap
- **Audio**: 500-word chunks with 100-word overlap
- **Images**: Full image + OCR text extraction

### Embedding Alignment
- Text: Sentence-BERT embeddings (384 dimensions)
- Images: CLIP embeddings projected to common space
- Cross-modal retrieval with dimension alignment

### Citation System
Every response includes:
- Numbered citations [1], [2], etc.
- Exact file paths and locations
- Page numbers for PDFs
- Timestamps for audio
- Metadata for images

## 🔒 Privacy & Security

- **Offline Operation**: No internet connection required
- **Local Processing**: All data stays on your USB/SSD
- **No External APIs**: Complete data sovereignty
- **USB-Friendly**: Optimized for portable drives

## 📊 Monitoring

### System Statistics
```bash
python main.py stats
```

**Web Interface:**
- Real-time index status
- Model availability
- Storage usage
- Processing statistics

### Health Checks
```bash
curl http://localhost:5000/health
```

## 🐛 Troubleshooting

### Common Issues

**"Model not loaded"**
- Ensure model file exists in `models/` directory
- Check model format (GGUF/GGML required)
- Verify llama-cpp-python installation

**"No dependencies found"**
```bash
python utils/install_check.py
# Follow the prompts to install missing packages
```

**"Index not found"**
- Run ingestion on your data first
- Check database directory permissions
- Verify FAISS installation

**"OCR not working"**
- Install Tesseract: `sudo apt install tesseract-ocr`
- Check image format support
- Verify pytesseract installation

### Performance Tuning

**For Large Datasets:**
```python
# Increase chunk size in ingesters
CHUNK_SIZE = 1200  # Default: 800

# Adjust search parameters
k = 10  # More results for better recall
```

**Memory Optimization:**
- Use smaller models for limited RAM
- Process files in smaller batches
- Enable GPU acceleration if available

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FAISS**: Meta's vector similarity search library
- **Sentence-BERT**: Compact text embeddings
- **CLIP**: OpenAI's vision-language model
- **Llama.cpp**: Local LLM inference
- **Whisper**: OpenAI's speech recognition

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues on GitHub
3. Create a new issue with detailed information

---

**Built with ❤️ for offline, privacy-focused multimodal AI**
