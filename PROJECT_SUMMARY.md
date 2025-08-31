# Project Summary: Generative AI Applications with RAG and LangChain

## 🎯 Project Overview

This repository contains the complete implementation of the "Generative AI Applications with RAG and LangChain" course assignment. All modules have been converted from Jupyter notebooks to Python scripts and optimized for local execution with GPU acceleration.

## 🚀 Key Features

- **Local Models Only**: No external API dependencies (IBM Watson removed)
- **GPU Optimized**: Full NVIDIA 4090 GPU support with CUDA acceleration
- **Complete RAG Pipeline**: End-to-end implementation from document loading to answer generation
- **Production Ready**: Clean, documented code with error handling
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 📚 Modules Completed

### ✅ Module 01: Document Processing Fundamentals
**Location**: `m01/`

**Files**:
- `full_document_retrieve_limitation.py` - Demonstrates LLM context length limitations
- `langchain_document_loader.py` - Various document loaders for different file formats
- `langchain_text_splitter_simple.py` - Text splitting techniques for RAG applications
- `requirements.txt` - Dependencies for Module 01
- `README.md` - Comprehensive documentation

**Key Concepts**:
- Document retrieval limitations due to LLM context length
- LangChain document loaders for various file formats
- Text splitting strategies for optimal chunking

### ✅ Module 02: Document Embedding and Vector Storage
**Location**: `m02/`

**Files**:
- `embed_documents_local_models.py` ⭐ **NEW** - Local embedding models with GPU optimization
- `langchain_retriever_local.py` ⭐ **NEW** - Various retriever types using local models
- `langchain_vector_store_local.py` ⭐ **NEW** - Vector databases with local models
- `requirements.txt` - Dependencies for Module 02
- `README.md` - Updated documentation for local models

**Key Concepts**:
- Local embedding models (Sentence Transformers, HuggingFace)
- Multiple retriever types (Vector Store, Multi-Query, Self-Querying, Parent Document)
- Vector databases (Chroma DB, FAISS) with local models

### ✅ Module 03: QA Bot Web Application
**Location**: `m03/`

**Files**:
- `qa_bot_local.py` ⭐ **NEW** - Complete QA Bot implementation
- `requirements.txt` - Dependencies for Module 03
- `README.md` - Comprehensive project documentation
- `markdown/` - Original project guides and requirements

**Key Concepts**:
- Complete RAG pipeline implementation
- Interactive Gradio web interface
- Local LLM integration (Ollama + HuggingFace fallback)
- PDF document processing and question answering

## 🛠️ Technical Implementation

### Local Model Architecture
```
Document → Loader → Splitter → Local Embeddings → Vector Store → Local LLM → Answer
    ↓         ↓        ↓           ↓              ↓            ↓        ↓
  PyPDF   Recursive  Sentence   Chroma DB    Retriever   RetrievalQA  Response
  Loader  TextSplit  Transformers            Similarity   Chain
```

### GPU Optimization Features
- **Automatic Detection**: CUDA availability checking
- **Memory Management**: Efficient GPU memory usage
- **Model Optimization**: GPU-optimized model loading
- **Fallback Support**: CPU execution if GPU unavailable

### Dependencies Management
- **Core**: LangChain ecosystem with compatible versions
- **Local Models**: Transformers, PyTorch, Sentence Transformers
- **Vector DBs**: Chroma DB, FAISS
- **Web Interface**: Gradio for interactive applications
- **Document Processing**: PyPDF, PyMuPDF

## 📋 Installation Instructions

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended for 4090)
- 8GB+ RAM, 10GB+ disk space

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/omare32/Project-Generative-AI-Applications-with-RAG-and-LangChain.git
cd Project-Generative-AI-Applications-with-RAG-and-LAG-and-LangChain

# Install core dependencies
pip install -r requirements.txt

# For GPU acceleration (recommended for 4090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optional: Install Ollama for local LLMs
# Visit https://ollama.ai/ for installation instructions
```

## 🚀 Usage Examples

### Module 01: Text Splitting
```bash
cd m01
python langchain_text_splitter_simple.py
```

### Module 02: Local Embeddings
```bash
cd m02
python embed_documents_local_models.py
```

### Module 03: QA Bot
```bash
cd m03
python qa_bot_local.py
# Open http://localhost:7860 in your browser
```

## 📸 Screenshots Required for Course

For the course assignment, capture screenshots of:

### Module 01
- Text splitting functionality
- Document loading demonstrations

### Module 02
- Local embedding model usage
- Vector database operations
- Retriever functionality

### Module 03
- PDF loader implementation
- Text splitter code
- Local embedding usage
- Vector database creation
- Retriever implementation
- Complete QA Bot interface

## 🔧 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### Model Download Issues
- Check internet connection
- Verify HuggingFace access
- Use smaller models for testing

#### Memory Issues
- Reduce chunk sizes
- Use smaller embedding models
- Monitor GPU memory with `nvidia-smi`

### Performance Tips
1. **GPU Memory**: Monitor usage and adjust chunk sizes
2. **Model Selection**: Choose appropriate sizes for your hardware
3. **Batch Processing**: Process multiple documents efficiently
4. **Caching**: Models are cached after first download

## 🎓 Learning Outcomes

By completing this project, you will have mastered:

1. **Document Processing**: Loading, splitting, and preprocessing various document formats
2. **Local AI Models**: Using and optimizing local embedding and language models
3. **Vector Databases**: Implementing efficient similarity search and retrieval
4. **RAG Architecture**: Building complete retrieval-augmented generation systems
5. **Web Applications**: Creating interactive AI applications with Gradio
6. **GPU Optimization**: Leveraging GPU acceleration for AI workloads

## 🚀 Next Steps

After completing this project:

1. **Customization**: Adapt the bot for your specific use cases
2. **Model Fine-tuning**: Customize models for your domain
3. **Performance Optimization**: Optimize for your specific hardware
4. **Production Deployment**: Deploy to production environments
5. **Advanced Features**: Add authentication, logging, and monitoring

## 📊 Project Statistics

- **Total Files**: 25+
- **Lines of Code**: 3,000+
- **Modules**: 3 complete
- **Dependencies**: 20+ packages
- **GPU Support**: Full NVIDIA 4090 optimization
- **Documentation**: Comprehensive READMEs for each module

## 🤝 Contributing

This project is part of the "Generative AI Applications with RAG and LangChain" course. Feel free to:

- Submit issues and enhancement requests
- Fork and modify for your own projects
- Share improvements with the community

## 📄 License

This project is part of the educational course and is provided for learning purposes.

## 🙏 Acknowledgments

- **Course Instructors**: For providing the comprehensive curriculum
- **LangChain Team**: For the excellent framework and documentation
- **HuggingFace**: For the open-source model ecosystem
- **Open Source Community**: For the tools and libraries that make this possible

---

**Status**: ✅ **COMPLETE** - All modules implemented and tested  
**Last Updated**: August 31, 2025  
**Version**: 1.0.0  
**GPU Support**: ✅ NVIDIA 4090 Optimized
