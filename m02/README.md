# Module 02: Document Embedding and Vector Storage (Local Models)

This module covers the essential concepts of document embedding and vector storage for RAG (Retrieval-Augmented Generation) applications using LangChain with local models optimized for GPU acceleration.

## Overview

Module 02 introduces three key concepts:
1. **Document Embedding** - Converting text to numerical vectors using local models (GPU optimized)
2. **Retrievers** - Various types of document retrieval mechanisms for finding relevant information
3. **Vector Stores** - Databases for storing and querying document embeddings efficiently

## Files

### 1. `embed_documents_local_models.py` ⭐ **NEW - GPU Optimized**
**Purpose**: Demonstrates how to use local embedding models optimized for GPU to embed documents.

**Key Concepts**:
- Document preprocessing and chunking for embedding
- Local embedding models with GPU acceleration (Sentence Transformers, HuggingFace)
- Query and document embedding generation
- Comparison of different local embedding approaches

**Features**:
- GPU detection and optimization for NVIDIA 4090
- Multiple embedding model implementations
- Custom transformers implementation
- Complete exercise solutions

### 2. `langchain_retriever_local.py` ⭐ **NEW - Local Models**
**Purpose**: Demonstrates various types of retrievers using local models for efficient document retrieval.

**Key Concepts**:
- Vector Store-backed Retriever for semantic similarity
- Multi-Query Retriever for comprehensive results
- Self-Querying Retriever for structured queries
- Parent Document Retriever for hierarchical retrieval

**Features**:
- Local embedding models (no external API calls)
- GPU acceleration support
- Custom retriever implementation
- Exercise solutions with practical examples

### 3. `langchain_vector_store_local.py` ⭐ **NEW - Local Models**
**Purpose**: Demonstrates vector databases for storing and querying document embeddings.

**Key Concepts**:
- Chroma DB for development and prototyping
- FAISS for large-scale similarity search
- Hybrid search combining multiple approaches
- Persistence and loading of vector stores

**Features**:
- Local embedding models with GPU support
- Performance comparison between vector stores
- Hybrid search implementation
- Custom similarity functions

### 4. `embed_documents_with_watsonx_embedding.py` (Legacy)
**Purpose**: Original IBM Watson implementation (kept for reference).

### 5. `langchain_retriever.py` (Legacy)
**Purpose**: Original IBM Watson implementation (kept for reference).

### 6. `langchain_vector_store.py` (Legacy)
**Purpose**: Original IBM Watson implementation (kept for reference).

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (recommended for 4090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Basic Usage

```bash
# Check GPU availability and run embedding demo
python embed_documents_local_models.py

# Test retriever functionality
python langchain_retriever_local.py

# Explore vector stores
python langchain_vector_store_local.py
```

### GPU Optimization

The scripts automatically detect and utilize your NVIDIA 4090 GPU:
- CUDA acceleration for embedding models
- GPU memory management
- Fallback to CPU if GPU unavailable

## Key Features

### 🚀 **Performance Optimized**
- GPU acceleration for embedding generation
- Efficient vector storage with Chroma DB and FAISS
- Optimized chunking strategies

### 🔒 **Privacy & Control**
- No external API calls required
- Local model execution
- Complete data sovereignty

### 🧠 **Advanced Retrieval**
- Multiple retriever types
- Hybrid search approaches
- Custom similarity functions

### 💾 **Persistence**
- Save and load vector stores
- Efficient storage formats
- Cross-session persistence

## Dependencies

- **LangChain**: Core framework for RAG applications
- **Sentence Transformers**: High-quality embedding models
- **Chroma DB**: Vector database for development
- **FAISS**: High-performance similarity search
- **PyTorch**: GPU acceleration and model support

## Exercises

Each script includes practical exercises:
1. **Custom Implementation**: Build your own retriever or similarity function
2. **Performance Comparison**: Compare different approaches
3. **Real-world Application**: Apply concepts to practical scenarios

## Troubleshooting

### GPU Issues
- Ensure CUDA is properly installed
- Check PyTorch CUDA version compatibility
- Verify GPU memory availability

### Model Download Issues
- Check internet connection
- Verify HuggingFace access
- Use smaller models for testing

### Memory Issues
- Reduce chunk sizes
- Use smaller embedding models
- Enable gradient checkpointing

## Next Steps

After completing Module 02:
1. **Module 03**: Advanced RAG techniques and applications
2. **Custom Projects**: Apply concepts to your own datasets
3. **Performance Tuning**: Optimize for your specific use case
4. **Model Fine-tuning**: Customize embedding models for your domain

## Contributing

Feel free to submit issues and enhancement requests!
