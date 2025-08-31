# Module 03: Build a QA Bot Web App (Local Models)

This module is the final project that integrates all the concepts learned in the course to build a complete QA Bot using RAG (Retrieval-Augmented Generation) and LangChain with local models optimized for GPU acceleration.

## Project Overview

**Build a QA Bot Web App** - This project simulates a real-world scenario where you build a bot that leverages LangChain and local large language models to answer questions based on content from loaded PDF documents.

## Learning Objectives

By completing this project, you will be able to:

- **Combine multiple components** to construct a fully functional QA bot
- **Leverage LangChain and local LLMs** to solve document-based question answering
- **Create an interactive web interface** using Gradio
- **Implement the complete RAG pipeline** from document loading to answer generation

## Project Components

The QA Bot implements the following key components:

### 1. **Document Loading** 📄
- Uses `PyPDFLoader` from LangChain to load PDF files
- Supports various PDF formats and sizes
- Automatic document parsing and text extraction

### 2. **Text Splitting** ✂️
- Implements `RecursiveCharacterTextSplitter` for intelligent chunking
- Configurable chunk size (1000 characters) and overlap (200 characters)
- Maintains document context while creating manageable pieces

### 3. **Local Embedding Models** 🧠
- Uses HuggingFace sentence transformers optimized for GPU
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings)
- Automatic GPU detection and CUDA acceleration for 4090

### 4. **Vector Database** 💾
- Chroma DB for storing document embeddings
- Efficient similarity search and retrieval
- Automatic collection management

### 5. **Intelligent Retrieval** 🔍
- Vector store-based retriever with similarity search
- Configurable number of results (k=3)
- Context-aware document retrieval

### 6. **Question-Answering Chain** ❓
- `RetrievalQA` chain from LangChain
- Combines retrieved documents with LLM reasoning
- Returns answers with source citations

### 7. **Web Interface** 🌐
- Gradio-based interactive web application
- File upload for PDFs
- Query input and answer display
- User-friendly design

## Files

### `qa_bot_local.py` ⭐ **Main Application**
**Purpose**: Complete QA Bot implementation using local models.

**Key Features**:
- GPU-optimized local models (no external API calls)
- Complete RAG pipeline implementation
- Interactive Gradio web interface
- Error handling and user feedback
- Source citation and transparency

### `requirements.txt`
**Purpose**: Lists all necessary dependencies for the QA Bot.

**Key Dependencies**:
- LangChain ecosystem
- Local model support (Transformers, PyTorch)
- Vector database (Chroma DB)
- Web interface (Gradio)
- PDF processing (PyPDF)

### `markdown/` 📚
**Purpose**: Original project documentation and guides.

**Contents**:
- `Project Overview.md` - Project requirements and objectives
- `Construct a QA Bot that Leverages LangChain and LLMs.md` - Step-by-step implementation guide
- `Set Up a Simple Gradio Interface to Interact with Your Models.md` - Gradio interface setup guide

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended for 4090)
- Sufficient disk space for models (~2-5 GB)

### Setup Steps

```bash
# 1. Navigate to Module 03
cd m03

# 2. Install dependencies
pip install -r requirements.txt

# 3. For GPU acceleration (recommended for 4090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Optional: Install Ollama for local LLMs
# Visit https://ollama.ai/ for installation instructions
```

## Usage

### Basic Usage

```bash
# Run the QA Bot
python qa_bot_local.py
```

The application will:
1. Check GPU availability and initialize models
2. Launch the Gradio web interface
3. Be available at `http://localhost:7860`

### Using the QA Bot

1. **Upload a PDF**: Use the file upload interface
2. **Ask a Question**: Type your question in the text box
3. **Get Answers**: Receive AI-generated answers with source citations
4. **Explore**: Try different questions and documents

### Example Queries

- "What is this document about?"
- "What are the main topics discussed?"
- "Can you summarize the key points?"
- "What are the conclusions?"
- "Explain the methodology used"

## Technical Architecture

```
PDF Document → Loader → Text Splitter → Embeddings → Vector Store → Retriever → QA Chain → LLM → Answer
     ↓              ↓           ↓           ↓           ↓           ↓         ↓        ↓
  PyPDFLoader  RecursiveChar  Sentence   Chroma DB  Similarity  RetrievalQA  Local   Response
                TextSplitter  Transformers           Search      Chain        LLM
```

## GPU Optimization

The QA Bot automatically detects and utilizes your NVIDIA 4090 GPU:

- **CUDA Acceleration**: Automatic GPU detection and utilization
- **Memory Management**: Efficient GPU memory usage
- **Model Optimization**: GPU-optimized model loading and inference
- **Fallback Support**: CPU fallback if GPU unavailable

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### Model Download Issues
```bash
# Check internet connection
# Verify HuggingFace access
# Use smaller models for testing
```

#### Memory Issues
- Reduce chunk sizes in `text_splitter()`
- Use smaller embedding models
- Enable gradient checkpointing

#### PDF Loading Issues
```bash
# Install PDF dependencies
pip install pypdf PyMuPDF
```

### Performance Tips

1. **GPU Memory**: Monitor GPU memory usage with `nvidia-smi`
2. **Chunk Size**: Adjust chunk sizes based on your GPU memory
3. **Model Selection**: Choose appropriate model sizes for your hardware
4. **Batch Processing**: Process multiple documents efficiently

## Project Requirements Completion

This implementation satisfies all project requirements:

✅ **Task 1**: Document loader using PyPDFLoader  
✅ **Task 2**: Text splitting with RecursiveCharacterTextSplitter  
✅ **Task 3**: Local embedding models (replaces IBM Watson)  
✅ **Task 4**: Vector database using Chroma DB  
✅ **Task 5**: Retriever with similarity search  
✅ **Task 6**: QA Bot with Gradio interface  

## Screenshots Required

For the course assignment, capture screenshots of:

1. **PDF Loader**: `pdf_loader.png` - Document loading functionality
2. **Text Splitter**: `code_splitter.png` - Text splitting implementation  
3. **Embedding**: `embedding.png` - Local embedding model usage
4. **Vector Database**: `vectordb.png` - Chroma DB creation
5. **Retriever**: `retriever.png` - Retriever implementation
6. **QA Bot Interface**: `qa_bot.png` - Complete Gradio interface

## Next Steps

After completing this project:

1. **Customization**: Adapt the bot for your specific use cases
2. **Model Fine-tuning**: Customize models for your domain
3. **Performance Optimization**: Optimize for your specific hardware
4. **Production Deployment**: Deploy to production environments
5. **Advanced Features**: Add authentication, logging, and monitoring

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is part of the "Generative AI Applications with RAG and LangChain" course.
