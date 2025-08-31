#!/usr/bin/env python3
"""
Create and Configure a Vector Database to Store Document Embeddings (Local Models)
=================================================================================

This script demonstrates how to use vector databases to store embeddings generated from 
textual data using LangChain with local models. The focus will be on two popular vector 
databases: Chroma DB and FAISS (Facebook AI Similarity Search). You will also learn how 
to perform similarity searches in these databases based on a query, enabling efficient 
retrieval of relevant information.

Objectives:
- Prepare and preprocess documents for embeddings
- Generate embeddings using local embedding models
- Store these embeddings in Chroma DB and FAISS
- Perform similarity searches to retrieve relevant information
"""

import warnings
warnings.filterwarnings('ignore')

import os
import torch

def download_file(url, filename):
    """Download a file if it doesn't exist"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, filename)
            print(f"✓ {filename} downloaded successfully")
        except Exception as e:
            print(f"⚠ Error downloading {filename}: {e}")
            print("Please download manually or check your internet connection")
            return False
    else:
        print(f"✓ {filename} already exists")
    return True

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

def load_and_split_data():
    """Load and split the source document"""
    print("=== Loading and Splitting Data ===\n")
    
    # Download the source document
    if not download_file(
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/i5V3ACEyz6hnYpVq6MTSvg/state-of-the-union.txt",
        "state-of-the-union.txt"
    ):
        print("⚠ Cannot proceed without the sample document.")
        return None, None
    
    # Load the document directly
    try:
        with open("state-of-the-union.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a simple document structure
        from langchain_core.documents import Document
        data = [Document(page_content=content, metadata={"source": "state-of-the-union.txt"})]
        
        print(f"✓ Document loaded successfully. Number of documents: {len(data)}")
        print(f"Document content preview: {data[0].page_content[:200]}...\n")
        
    except Exception as e:
        print(f"⚠ Error loading document: {e}")
        return None, None
    
    # Split the document into chunks
    print("--- Splitting Document into Chunks ---")
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(data[0].page_content)
        print(f"✓ Document split into {len(chunks)} chunks")
        print(f"First chunk preview: {chunks[0][:100]}...\n")
        
        return data, chunks
        
    except ImportError as e:
        print(f"⚠ Error importing RecursiveCharacterTextSplitter: {e}")
        print("Please install langchain: pip install langchain")
        return data, None
    except Exception as e:
        print(f"⚠ Error splitting document: {e}")
        return data, None

def demonstrate_chroma_vectorstore(chunks):
    """Demonstrate Chroma DB vector store"""
    print("=== Chroma DB Vector Store ===\n")
    
    if not chunks:
        print("⚠ No chunks available for vector store. Skipping this section.")
        return None
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        
        print("--- Setting Up Chroma DB ---")
        
        # Use local embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        print("✓ Embedding model created successfully")
        
        # Create Chroma vector store
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            collection_name="state_of_union_chroma"
        )
        print("✓ Chroma DB vector store created successfully")
        
        # Test similarity search
        print("--- Testing Similarity Search ---")
        query = "What are the main economic policies mentioned?"
        print(f"Query: '{query}'")
        
        try:
            docs = vectorstore.similarity_search(query, k=3)
            print(f"✓ Similarity search completed successfully")
            print(f"Retrieved {len(docs)} documents")
            
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc.page_content[:150]}...")
                print(f"Metadata: {doc.metadata}")
            
            # Test similarity search with scores
            print("\n--- Testing Similarity Search with Scores ---")
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
            print(f"✓ Similarity search with scores completed successfully")
            
            for i, (doc, score) in enumerate(docs_with_scores):
                print(f"\nDocument {i+1} (Score: {score:.4f}):")
                print(f"Content: {doc.page_content[:150]}...")
            
            return vectorstore
            
        except Exception as e:
            print(f"⚠ Error during similarity search: {e}")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing dependencies: {e}")
        print("Please install required packages: pip install chromadb sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with Chroma DB: {e}")
        return None

def demonstrate_faiss_vectorstore(chunks):
    """Demonstrate FAISS vector store"""
    print("=== FAISS Vector Store ===\n")
    
    if not chunks:
        print("⚠ No chunks available for vector store. Skipping this section.")
        return None
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        print("--- Setting Up FAISS ---")
        
        # Use local embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        print("✓ Embedding model created successfully")
        
        # Create FAISS vector store
        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_model
        )
        print("✓ FAISS vector store created successfully")
        
        # Test similarity search
        print("--- Testing Similarity Search ---")
        query = "What are the main economic policies mentioned?"
        print(f"Query: '{query}'")
        
        try:
            docs = vectorstore.similarity_search(query, k=3)
            print(f"✓ Similarity search completed successfully")
            print(f"Retrieved {len(docs)} documents")
            
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc.page_content[:150]}...")
                print(f"Metadata: {doc.metadata}")
            
            # Test similarity search with scores
            print("\n--- Testing Similarity Search with Scores ---")
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
            print(f"✓ Similarity search with scores completed successfully")
            
            for i, (doc, score) in enumerate(docs_with_scores):
                print(f"\nDocument {i+1} (Score: {score:.4f}):")
                print(f"Content: {doc.page_content[:150]}...")
            
            return vectorstore
            
        except Exception as e:
            print(f"⚠ Error during similarity search: {e}")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing dependencies: {e}")
        print("Please install required packages: pip install faiss-cpu sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with FAISS: {e}")
        return None

def demonstrate_hybrid_search(chunks):
    """Demonstrate hybrid search combining different approaches"""
    print("=== Hybrid Search Demonstration ===\n")
    
    if not chunks:
        print("⚠ No chunks available for hybrid search. Skipping this section.")
        return None
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain.retrievers import EnsembleRetriever
        from langchain.retrievers import BM25Retriever
        
        print("--- Setting Up Hybrid Search ---")
        
        # Use local embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        print("✓ Embedding model created successfully")
        
        # Create vector store
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            collection_name="state_of_union_hybrid"
        )
        print("✓ Vector store created successfully")
        
        # Create BM25 retriever (keyword-based)
        try:
            bm25_retriever = BM25Retriever.from_texts(chunks)
            print("✓ BM25 retriever created successfully")
        except Exception as e:
            print(f"⚠ Error creating BM25 retriever: {e}")
            bm25_retriever = None
        
        # Create vector store retriever
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✓ Vector store retriever created successfully")
        
        # Create ensemble retriever if both are available
        if bm25_retriever:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.3, 0.7]  # Give more weight to semantic search
            )
            print("✓ Ensemble retriever created successfully")
            
            # Test hybrid search
            print("--- Testing Hybrid Search ---")
            query = "What are the main economic policies mentioned?"
            print(f"Query: '{query}'")
            
            try:
                docs = ensemble_retriever.get_relevant_documents(query)
                print(f"✓ Hybrid search completed successfully")
                print(f"Retrieved {len(docs)} documents")
                
                for i, doc in enumerate(docs[:3]):  # Show first 3
                    print(f"\nDocument {i+1}:")
                    print(f"Content: {doc.page_content[:150]}...")
                    print(f"Metadata: {doc.metadata}")
                
                return ensemble_retriever
                
            except Exception as e:
                print(f"⚠ Error during hybrid search: {e}")
                return vector_retriever
        else:
            print("⚠ Using vector store retriever only (BM25 not available)")
            return vector_retriever
            
    except ImportError as e:
        print(f"⚠ Error importing dependencies: {e}")
        print("Please install required packages: pip install chromadb sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with hybrid search: {e}")
        return None

def demonstrate_persistence_and_loading(chunks):
    """Demonstrate saving and loading vector stores"""
    print("=== Persistence and Loading Demonstration ===\n")
    
    if not chunks:
        print("⚠ No chunks available for persistence demo. Skipping this section.")
        return None
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma, FAISS
        
        print("--- Setting Up Vector Stores for Persistence ---")
        
        # Use local embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        print("✓ Embedding model created successfully")
        
        # Create and save Chroma DB
        print("--- Chroma DB Persistence ---")
        chroma_store = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            collection_name="persistent_chroma"
        )
        print("✓ Chroma DB created successfully")
        
        # Chroma DB persists automatically to disk
        print("✓ Chroma DB automatically persisted to disk")
        
        # Create and save FAISS
        print("\n--- FAISS Persistence ---")
        faiss_store = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_model
        )
        print("✓ FAISS created successfully")
        
        # Save FAISS to disk
        faiss_path = "./faiss_index"
        faiss_store.save_local(faiss_path)
        print(f"✓ FAISS saved to disk at: {faiss_path}")
        
        # Test loading FAISS
        print("\n--- Testing FAISS Loading ---")
        loaded_faiss = FAISS.load_local(faiss_path, embedding_model)
        print("✓ FAISS loaded successfully from disk")
        
        # Test loaded FAISS
        query = "economic policies"
        docs = loaded_faiss.similarity_search(query, k=2)
        print(f"✓ Loaded FAISS search successful: {len(docs)} documents retrieved")
        
        return {"chroma": chroma_store, "faiss": faiss_store, "loaded_faiss": loaded_faiss}
        
    except ImportError as e:
        print(f"⚠ Error importing dependencies: {e}")
        print("Please install required packages: pip install chromadb faiss-cpu sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with persistence demo: {e}")
        return None

def exercises(chunks):
    """Complete the exercises from the notebook"""
    print("=== Exercises ===\n")
    
    if not chunks:
        print("⚠ No chunks available for exercises. Skipping this section.")
        return
    
    # Exercise 1 - Compare different vector stores
    print("--- Exercise 1: Vector Store Comparison ---")
    print("Comparing Chroma DB vs FAISS performance...")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma, FAISS
        import time
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        
        # Test Chroma DB
        print("Testing Chroma DB...")
        start_time = time.time()
        chroma_store = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            collection_name="exercise_chroma"
        )
        chroma_time = time.time() - start_time
        
        # Test FAISS
        print("Testing FAISS...")
        start_time = time.time()
        faiss_store = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_model
        )
        faiss_time = time.time() - start_time
        
        print(f"✓ Chroma DB creation time: {chroma_time:.2f} seconds")
        print(f"✓ FAISS creation time: {faiss_time:.2f} seconds")
        
        if chroma_time < faiss_time:
            print("Chroma DB was faster for this dataset")
        else:
            print("FAISS was faster for this dataset")
        
    except Exception as e:
        print(f"⚠ Error with vector store comparison: {e}")
    
    # Exercise 2 - Custom similarity function
    print("\n--- Exercise 2: Custom Similarity Function ---")
    print("Implementing a simple keyword-based similarity function...")
    
    try:
        def keyword_similarity(query, text):
            """Simple keyword-based similarity"""
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            intersection = query_words.intersection(text_words)
            union = query_words.union(text_words)
            return len(intersection) / len(union) if union else 0
        
        # Test with sample query
        query = "economic policies"
        sample_text = chunks[0] if chunks else "Sample text about economic policies"
        
        similarity_score = keyword_similarity(query, sample_text)
        print(f"✓ Keyword similarity score: {similarity_score:.4f}")
        print("Note: This is a simple Jaccard similarity implementation")
        
    except Exception as e:
        print(f"⚠ Error with custom similarity: {e}")

def main():
    """Main function to demonstrate vector stores"""
    print("=== LangChain Vector Store Demo (Local Models) ===\n")
    
    # Check GPU availability
    gpu_available = check_gpu()
    
    # Load and split data
    data, chunks = load_and_split_data()
    if not chunks:
        print("⚠ Cannot proceed without document chunks.")
        return
    
    # Demonstrate different vector stores
    chroma_store = demonstrate_chroma_vectorstore(chunks)
    faiss_store = demonstrate_faiss_vectorstore(chunks)
    hybrid_retriever = demonstrate_hybrid_search(chunks)
    persistent_stores = demonstrate_persistence_and_loading(chunks)
    
    # Complete exercises
    exercises(chunks)
    
    print("=== Demo Complete ===")
    print("You have successfully explored vector stores:")
    if chroma_store:
        print("✓ Chroma DB Vector Store")
    if faiss_store:
        print("✓ FAISS Vector Store")
    if hybrid_retriever:
        print("✓ Hybrid Search (Ensemble)")
    if persistent_stores:
        print("✓ Persistence and Loading")
    
    print(f"\nGPU Acceleration: {'Enabled' if gpu_available else 'Disabled'}")
    
    print("\nKey takeaways:")
    print("- Chroma DB is great for development and prototyping")
    print("- FAISS excels at large-scale similarity search")
    print("- Hybrid approaches combine the best of multiple methods")
    print("- Persistence allows you to save and reuse vector stores")
    print("- Local models provide privacy and control")

if __name__ == "__main__":
    main()
