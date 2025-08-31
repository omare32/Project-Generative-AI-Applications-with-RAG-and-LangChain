# Module 02: Document Embedding and Vector Storage

This module covers the essential concepts of document embedding and vector storage for RAG (Retrieval-Augmented Generation) applications using LangChain.

## Overview

Module 02 introduces three key concepts:
1. **Document Embedding** - Converting text to numerical vectors using Watsonx and HuggingFace models
2. **Retrievers** - Various types of document retrieval mechanisms for finding relevant information
3. **Vector Stores** - Databases for storing and querying document embeddings efficiently

## Files

### 1. `embed_documents_with_watsonx_embedding.py`
**Purpose**: Demonstrates how to use embedding models from watsonx.ai and Hugging Face to embed documents.

**Key Concepts**:
- Document preprocessing and chunking for embedding
- IBM Watsonx.ai embedding models (slate-125m-english-rtrvr)
- HuggingFace sentence-transformers (all-mpnet-base-v2)
- Query and document embedding generation
- Embedding dimension analysis (768-dimensional vectors)

**Learning Objectives**:
- Prepare and preprocess documents for embedding
- Use watsonx.ai and Hugging Face embedding models to generate embeddings

### 2. `langchain_retriever.py`
**Purpose**: Demonstrates how to use various types of retrievers to efficiently extract relevant document segments from text using LangChain.

**Retrievers Covered**:
- **Vector Store-Backed Retriever**: Basic similarity search with various search types (similarity, MMR, threshold)
- **Multi-Query Retriever**: Generates multiple query variations for comprehensive results
- **Self-Querying Retriever**: Automatically constructs structured queries from natural language
- **Parent Document Retriever**: Balances small chunks for accuracy with large chunks for context

**Key Features**:
- Similarity search with configurable parameters
- MMR (Maximum Marginal Relevance) for diverse results
- Metadata-based filtering and querying
- Context-aware document retrieval

**Learning Objectives**:
- Use various types of retrievers for efficient document extraction
- Apply different search strategies for optimal results
- Implement intelligent query processing and filtering

### 3. `langchain_vector_store.py`
**Purpose**: Demonstrates how to use vector databases to store embeddings generated from textual data using LangChain.

**Vector Databases Covered**:
- **Chroma DB**: Easy-to-use vector database with good performance
- **FAISS**: High-performance vector similarity search library
- **Vector Store Management**: CRUD operations (Create, Read, Update, Delete)

**Key Features**:
- Document embedding storage and retrieval
- Similarity search with configurable parameters
- Database management operations
- Performance comparison between different vector stores

**Learning Objectives**:
- Prepare and preprocess documents for embeddings
- Generate embeddings using watsonx.ai's embedding model
- Store embeddings in Chroma DB and FAISS
- Perform similarity searches for relevant document retrieval

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up IBM Watson AI credentials** (for embedding functionality):
   - You'll need access to IBM's watsonx.ai platform
   - Set up your project ID and credentials

## Usage

### Running Individual Scripts

```bash
# Run the document embedding demo
python embed_documents_with_watsonx_embedding.py

# Run the retriever demo
python langchain_retriever.py

# Run the vector store demo
python langchain_vector_store.py
```

### Running All Scripts

```bash
# Run all demonstrations in sequence
python embed_documents_with_watsonx_embedding.py
python langchain_retriever.py
python langchain_vector_store.py
```

## Key Concepts

### Document Embedding
- **Purpose**: Convert text to numerical vectors that capture semantic meaning
- **Models**: IBM Watsonx.ai (slate-125m-english-rtrvr) and HuggingFace (all-mpnet-base-v2)
- **Dimensions**: 768-dimensional vectors for semantic representation
- **Applications**: Similarity search, clustering, classification

### Retrievers
- **Vector Store-Backed**: Basic similarity search with configurable search types
- **Multi-Query**: Generate multiple query variations for comprehensive coverage
- **Self-Querying**: Natural language to structured query conversion
- **Parent Document**: Balance accuracy and context in document retrieval

### Vector Stores
- **Chroma DB**: User-friendly vector database with good performance
- **FAISS**: High-performance similarity search library
- **Operations**: Store, retrieve, update, and delete embeddings
- **Search**: Similarity-based document retrieval

## Workflow

1. **Document Loading**: Load source documents using appropriate loaders
2. **Text Splitting**: Split documents into manageable chunks
3. **Embedding Generation**: Convert text chunks to numerical vectors
4. **Vector Storage**: Store embeddings in vector databases
5. **Similarity Search**: Retrieve relevant documents based on queries
6. **Advanced Retrieval**: Use specialized retrievers for complex scenarios

## Next Steps

After completing this module, you'll be ready to:
- Move to Module 03: Building complete RAG applications
- Implement document embedding pipelines in production
- Design efficient retrieval systems for large document collections
- Optimize vector storage and search performance

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed from `requirements.txt`
- **Embedding Errors**: Check your IBM Watson AI credentials and project setup
- **Vector Store Issues**: Verify ChromaDB and FAISS installations
- **Memory Issues**: Large document collections may require significant memory

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [IBM Watson AI Documentation](https://ibm.github.io/watsonx-ai-python-sdk/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
