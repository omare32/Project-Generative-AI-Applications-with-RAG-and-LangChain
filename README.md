# 🤖 Generative AI Applications with RAG and LangChain

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.11-green.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%204090-purple.svg)](https://www.nvidia.com/)

> **A comprehensive course project demonstrating Retrieval-Augmented Generation (RAG) using LangChain and local AI models, optimized for GPU acceleration.**

## 🎯 Project Overview

This project implements a complete **QA Bot Web Application** that leverages LangChain and Large Language Models (LLMs) to answer questions based on content from loaded PDF documents. The system uses RAG (Retrieval-Augmented Generation) architecture with local, open-source models optimized for GPU acceleration.

### ✨ Key Features

- 🚀 **Complete RAG Pipeline**: Document loading → Text splitting → Embedding → Vector storage → Retrieval → Generation
- 🖥️ **GPU Optimization**: CUDA support for NVIDIA 4090 GPU with PyTorch acceleration
- 🏠 **Local Models**: No external API dependencies, completely self-contained
- 🎨 **Beautiful UI**: Modern Gradio web interface with responsive design
- 📚 **Multi-Format Support**: PDF document processing with intelligent chunking
- 🔍 **Smart Retrieval**: Similarity search using Chroma vector database
- 📱 **Web Interface**: Upload PDFs and ask questions through an intuitive web UI

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Document  │───▶│  Document Loader│───▶│  Text Splitter  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vector Store  │◀───│   Embeddings    │◀───│   Text Chunks   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Retriever    │───▶│   QA Chain      │───▶│   LLM Response  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
├── 📁 m01/                          # Module 01: Document Processing
│   ├── 📄 langchain_document_loader.py
│   ├── 📄 langchain_text_splitter.py
│   └── 📄 full_document_retrieve_limitation.py
│
├── 📁 m02/                          # Module 02: Embeddings & Vector DB
│   ├── 📄 embed_documents_local_models.py
│   ├── 📄 langchain_retriever_local.py
│   └── 📄 langchain_vector_store_local.py
│
├── 📁 m03/                          # Module 03: QA Bot Implementation
│   ├── 📄 qa_bot_local.py
│   ├── 📄 generate_submission_pngs.py
│   ├── 📄 document.to.load.pdf      # Sample PDF for testing
│   └── 📁 markdown/                 # Project documentation
│
├── 🚀 main_qa_bot.py                # Main integrated QA bot
├── 📋 requirements.txt               # Project dependencies
├── 📖 README.md                     # This file
└── 📊 PROJECT_SUMMARY.md            # Comprehensive project overview
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **NVIDIA GPU** (recommended for optimal performance)
- **CUDA Toolkit** (for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/omare32/Project-Generative-AI-Applications-with-RAG-and-LangChain.git
   cd Project-Generative-AI-Applications-with-RAG-and-LangChain
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main QA bot**
   ```bash
   python main_qa_bot.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:7860`
   - Upload any PDF document
   - Start asking questions!

## 🎮 Usage

### Basic Usage

1. **Upload PDF**: Drag and drop any PDF file into the upload area
2. **Ask Questions**: Type your question in the query box
3. **Get Answers**: Click "Get Answer" to receive AI-generated responses

### Example Questions

- "What is this document about?"
- "What are the main topics discussed?"
- "Can you summarize the key points?"
- "What are the conclusions of this paper?"
- "Explain the methodology used in this research"

### Advanced Features

- **GPU Acceleration**: Automatically detects and uses NVIDIA GPU
- **Smart Chunking**: Intelligent document splitting for optimal retrieval
- **Vector Search**: Semantic similarity search using embeddings
- **Local Processing**: No data leaves your machine

## 🔧 Configuration

### GPU Settings

The system automatically detects GPU availability:
- **CUDA**: Automatically uses NVIDIA GPU if available
- **CPU Fallback**: Gracefully falls back to CPU if no GPU detected

### Model Configuration

```python
# Embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings
chunk_size = 1000
chunk_overlap = 200

# Retrieval settings
k = 4  # Number of relevant chunks to retrieve
```

## 📊 Performance

### GPU Acceleration

- **NVIDIA 4090**: ~10-50x faster than CPU for embeddings
- **Memory Usage**: Optimized for 24GB VRAM
- **Batch Processing**: Efficient parallel processing

### Benchmarks

| Model | Device | Embedding Speed | Memory Usage |
|-------|--------|-----------------|--------------|
| all-MiniLM-L6-v2 | GPU (4090) | ~1000 docs/sec | ~2GB VRAM |
| all-MiniLM-L6-v2 | CPU | ~50 docs/sec | ~4GB RAM |

## 🧪 Testing

### Sample PDF

Use the included `m03/document.to.load.pdf` for testing:
```bash
# Test with sample document
python main_qa_bot.py
# Upload: m03/document.to.load.pdf
# Query: "What is this paper about?"
```

### Unit Tests

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=.
```

## 📚 Learning Modules

### Module 01: Document Processing
- **Document Loaders**: PyPDFLoader, TextLoader, and more
- **Text Splitting**: RecursiveCharacterTextSplitter strategies
- **Context Limitations**: Understanding LLM context windows

### Module 02: Embeddings & Vector DB
- **Local Embeddings**: HuggingFace sentence-transformers
- **Vector Stores**: Chroma DB and FAISS integration
- **Retrieval Methods**: Similarity search and filtering

### Module 03: QA Bot Implementation
- **RAG Pipeline**: Complete question-answering system
- **Gradio Interface**: Beautiful web UI
- **Local LLMs**: Ollama and HuggingFace integration

## 🛠️ Development

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Adding New Features

1. **Fork the repository**
2. **Create a feature branch**
3. **Implement your changes**
4. **Add tests**
5. **Submit a pull request**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution

- 🚀 **Performance optimization**
- 🎨 **UI/UX improvements**
- 🔧 **Additional document formats**
- 🌐 **Multi-language support**
- 📊 **Analytics and monitoring**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain Team** for the amazing framework
- **HuggingFace** for open-source models
- **Chroma DB** for vector storage
- **Gradio** for the beautiful web interface

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/omare32/Project-Generative-AI-Applications-with-RAG-and-LangChain/issues)
- **Discussions**: [GitHub Discussions](https://github.com/omare32/Project-Generative-AI-Applications-with-RAG-and-LangChain/discussions)
- **Wiki**: [Project Wiki](https://github.com/omare32/Project-Generative-AI-Applications-with-RAG-and-LangChain/wiki)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=omare32/Project-Generative-AI-Applications-with-RAG-and-LangChain&type=Date)](https://star-history.com/#omare32/Project-Generative-AI-Applications-with-RAG-and-LangChain&Date)

---

<div align="center">

**Made with ❤️ for the AI community**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.11-green.svg)](https://langchain.com/)

</div>
