#!/usr/bin/env python3
"""
QA Bot that Leverages LangChain and Local LLMs (GPU Optimized)
==============================================================

This script constructs a question-answering (QA) bot that leverages LangChain and local large language models 
to answer questions based on content from loaded PDF documents. The bot combines various components including 
document loaders, text splitters, embedding models, vector databases, retrievers, and Gradio as the front-end interface.

Since you have a 4090 GPU, this implementation uses local models for optimal performance and privacy.

Learning objectives:
- Combine multiple components to construct a fully functional QA bot
- Leverage LangChain and local LLMs to solve document-based question answering
- Create an interactive web interface using Gradio
"""

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import gradio as gr
from typing import List, Dict, Any

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# Local model imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def check_gpu():
    """Check GPU availability and capabilities"""
    print("=== GPU Information ===\n")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Count: {gpu_count}")
        print(f"✓ Current Device: {current_device}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        print(f"✓ Memory Allocated: {memory_allocated:.2f} GB")
        print(f"✓ Memory Reserved: {memory_reserved:.2f} GB\n")
        
        return True
    else:
        print("⚠ No GPU available. Using CPU instead.\n")
        return False

def get_llm():
    """
    Initialize and return a local LLM.
    Tries Ollama first, then falls back to HuggingFace models.
    """
    print("=== Initializing Local LLM ===\n")
    
    # Try Ollama first (if available)
    try:
        print("Attempting to use Ollama...")
        llm = Ollama(model="llama2")
        print("✓ Ollama LLM initialized successfully")
        return llm
    except Exception as e:
        print(f"⚠ Ollama not available: {e}")
        print("Falling back to HuggingFace model...")
    
    # Fallback to HuggingFace model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Use a smaller, efficient model
        model_name = "microsoft/DialoGPT-medium"  # Smaller model for testing
        
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if device == "cuda":
            model = model.to(device)
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        print("✓ HuggingFace LLM initialized successfully")
        return llm
        
    except Exception as e:
        print(f"⚠ Error initializing HuggingFace model: {e}")
        print("⚠ Cannot proceed without LLM. Please install Ollama or check your setup.")
        return None

def document_loader(file):
    """
    Load PDF documents using LangChain's PyPDFLoader.
    
    Args:
        file: Gradio file object with PDF file
        
    Returns:
        List of loaded documents
    """
    print("=== Loading PDF Document ===\n")
    
    try:
        if not file or not file.name:
            print("⚠ No file provided")
            return None
        
        print(f"Loading file: {file.name}")
        
        # Create PDF loader
        loader = PyPDFLoader(file.name)
        
        # Load the document
        loaded_document = loader.load()
        
        print(f"✓ Document loaded successfully")
        print(f"Number of pages: {len(loaded_document)}")
        print(f"First page preview: {loaded_document[0].page_content[:200]}...\n")
        
        return loaded_document
        
    except ImportError as e:
        print(f"⚠ Error importing PyPDFLoader: {e}")
        print("Please install pypdf: pip install pypdf")
        return None
    except Exception as e:
        print(f"⚠ Error loading document: {e}")
        return None

def text_splitter(data):
    """
    Split documents into manageable chunks using RecursiveCharacterTextSplitter.
    
    Args:
        data: List of loaded documents
        
    Returns:
        List of text chunks
    """
    print("=== Splitting Document into Chunks ===\n")
    
    if not data:
        print("⚠ No data provided for splitting")
        return None
    
    try:
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Split the documents
        chunks = text_splitter.split_documents(data)
        
        print(f"✓ Document split successfully")
        print(f"Number of chunks: {len(chunks)}")
        print(f"First chunk preview: {chunks[0].page_content[:200]}...\n")
        
        return chunks
        
    except ImportError as e:
        print(f"⚠ Error importing RecursiveCharacterTextSplitter: {e}")
        print("Please install langchain: pip install langchain")
        return None
    except Exception as e:
        print(f"⚠ Error splitting document: {e}")
        return None

def local_embedding():
    """
    Initialize and return a local embedding model.
    Uses HuggingFace sentence transformers optimized for GPU.
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    print("=== Initializing Local Embedding Model ===\n")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Use a lightweight but effective model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading embedding model: {model_name}")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        
        print("✓ Local embedding model initialized successfully")
        
        # Test the embedding model
        test_text = "This is a test sentence."
        test_embedding = embedding_model.embed_query(test_text)
        print(f"✓ Test embedding generated successfully")
        print(f"Embedding dimension: {len(test_embedding)}")
        print(f"First 5 values: {test_embedding[:5]}\n")
        
        return embedding_model
        
    except ImportError as e:
        print(f"⚠ Error importing HuggingFaceEmbeddings: {e}")
        print("Please install sentence-transformers: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error initializing embedding model: {e}")
        return None

def vector_database(chunks):
    """
    Create a vector database to store document embeddings.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Chroma vector store instance
    """
    print("=== Creating Vector Database ===\n")
    
    if not chunks:
        print("⚠ No chunks provided for vector database")
        return None
    
    try:
        # Get embedding model
        embedding_model = local_embedding()
        if not embedding_model:
            print("⚠ Cannot create vector database without embedding model")
            return None
        
        # Create Chroma vector store
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="qa_bot_documents"
        )
        
        print("✓ Vector database created successfully")
        print(f"Collection name: qa_bot_documents")
        print(f"Number of documents stored: {len(chunks)}\n")
        
        return vectordb
        
    except ImportError as e:
        print(f"⚠ Error importing Chroma: {e}")
        print("Please install chromadb: pip install chromadb")
        return None
    except Exception as e:
        print(f"⚠ Error creating vector database: {e}")
        return None

def retriever(file):
    """
    Create a retriever that loads, splits, embeds, and converts documents.
    
    Args:
        file: Gradio file object with PDF file
        
    Returns:
        Retriever instance
    """
    print("=== Creating Document Retriever ===\n")
    
    try:
        # Load document
        splits = document_loader(file)
        if not splits:
            print("⚠ Cannot create retriever without loaded documents")
            return None
        
        # Split into chunks
        chunks = text_splitter(splits)
        if not chunks:
            print("⚠ Cannot create retriever without document chunks")
            return None
        
        # Create vector database
        vectordb = vector_database(chunks)
        if not vectordb:
            print("⚠ Cannot create retriever without vector database")
            return None
        
        # Create retriever
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        print("✓ Document retriever created successfully")
        print(f"Search type: similarity")
        print(f"Number of results: 3\n")
        
        return retriever
        
    except Exception as e:
        print(f"⚠ Error creating retriever: {e}")
        return None

def retriever_qa(file, query):
    """
    Perform question-answering using the RetrievalQA chain.
    
    Args:
        file: Gradio file object with PDF file
        query: User's question
        
    Returns:
        Answer to the question
    """
    print("=== Processing Question-Answering Query ===\n")
    
    if not query or not query.strip():
        return "Please enter a question."
    
    print(f"Query: {query}")
    
    try:
        # Get LLM
        llm = get_llm()
        if not llm:
            return "Error: LLM not available. Please check your setup."
        
        # Get retriever
        retriever_obj = retriever(file)
        if not retriever_obj:
            return "Error: Retriever not available. Please check your document."
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=True
        )
        
        print("✓ QA chain created successfully")
        print("Processing query...")
        
        # Process the query
        response = qa.invoke({"query": query})
        
        print("✓ Query processed successfully")
        
        # Extract answer and sources
        answer = response.get('result', 'No answer generated')
        source_docs = response.get('source_documents', [])
        
        # Format response with sources
        formatted_answer = f"Answer: {answer}\n\n"
        if source_docs:
            formatted_answer += "Sources:\n"
            for i, doc in enumerate(source_docs[:2]):  # Show first 2 sources
                source_text = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                formatted_answer += f"{i+1}. {source_text}\n"
        
        return formatted_answer
        
    except Exception as e:
        print(f"⚠ Error in QA processing: {e}")
        return f"Error processing your question: {str(e)}"

def create_gradio_interface():
    """Create and return the Gradio interface"""
    print("=== Creating Gradio Interface ===\n")
    
    # Create Gradio interface
    rag_application = gr.Interface(
        fn=retriever_qa,
        allow_flagging="never",
        inputs=[
            gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
            gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
        ],
        outputs=gr.Textbox(label="Answer", lines=10),
        title="Local RAG QA Bot",
        description="Upload a PDF document and ask any question. The chatbot will answer using the provided document content.",
        examples=[
            ["What is this document about?", "Upload a PDF first"],
            ["What are the main topics discussed?", "Upload a PDF first"],
            ["Can you summarize the key points?", "Upload a PDF first"]
        ]
    )
    
    print("✓ Gradio interface created successfully")
    return rag_application

def main():
    """Main function to run the QA Bot"""
    print("=== Local RAG QA Bot (GPU Optimized) ===\n")
    
    # Check GPU availability
    gpu_available = check_gpu()
    
    # Test LLM availability
    test_llm = get_llm()
    if not test_llm:
        print("⚠ Critical Error: LLM not available")
        print("Please install Ollama or ensure HuggingFace models are accessible")
        return
    
    # Test embedding model
    test_embedding = local_embedding()
    if not test_embedding:
        print("⚠ Critical Error: Embedding model not available")
        print("Please install sentence-transformers")
        return
    
    print("✓ All components available. Creating Gradio interface...")
    
    # Create and launch the interface
    rag_application = create_gradio_interface()
    
    print("\n=== Launching Application ===")
    print("The QA Bot will be available at: http://localhost:7860")
    print("Press Ctrl+C to stop the application")
    
    # Launch the application
    rag_application.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True
    )

if __name__ == "__main__":
    main()
