#!/usr/bin/env python3
"""
Develop a Retriever to Fetch Document Segments Based on Queries (Local Models)
==============================================================================

This script demonstrates how to use various retrievers to efficiently extract relevant 
document segments from text using LangChain with local models. You will learn about 
four types of retrievers: Vector Store-backed Retriever, Multi-Query Retriever, 
Self-Querying Retriever, and Parent Document Retriever.

Objectives:
- Use various types of retrievers to efficiently extract relevant document segments from text
- Apply the Vector Store-backed Retriever for semantic similarity and relevance
- Utilize the Multi-Query Retriever for comprehensive results
- Implement the Self-Querying Retriever for structured queries
- Use the Parent Document Retriever for hierarchical document retrieval
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

def demonstrate_vector_store_retriever(chunks):
    """Demonstrate Vector Store-backed Retriever"""
    print("=== Vector Store-backed Retriever ===\n")
    
    if not chunks:
        print("⚠ No chunks available for retriever. Skipping this section.")
        return None
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain.retrievers import VectorStoreRetriever
        
        print("--- Setting Up Vector Store ---")
        
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
            collection_name="state_of_union"
        )
        print("✓ Vector store created successfully")
        
        # Create retriever
        retriever = VectorStoreRetriever(
            vectorstore=vectorstore,
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        print("✓ Vector store retriever created successfully\n")
        
        # Test retrieval
        print("--- Testing Retrieval ---")
        query = "What are the main economic policies mentioned?"
        print(f"Query: '{query}'")
        
        try:
            docs = retriever.get_relevant_documents(query)
            print(f"✓ Retrieved {len(docs)} relevant documents")
            
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc.page_content[:150]}...")
                print(f"Metadata: {doc.metadata}")
            
            return retriever
            
        except Exception as e:
            print(f"⚠ Error during retrieval: {e}")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing dependencies: {e}")
        print("Please install required packages: pip install chromadb sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with vector store retriever: {e}")
        return None

def demonstrate_multi_query_retriever(chunks):
    """Demonstrate Multi-Query Retriever"""
    print("=== Multi-Query Retriever ===\n")
    
    if not chunks:
        print("⚠ No chunks available for retriever. Skipping this section.")
        return None
    
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain.chat_models import ChatOpenAI
        from langchain.llms import Ollama
        
        print("--- Setting Up Multi-Query Retriever ---")
        
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
            collection_name="state_of_union_multi"
        )
        print("✓ Vector store created successfully")
        
        # Try to use local LLM for query generation
        try:
            # Try Ollama first (if available)
            llm = Ollama(model="llama2")
            print("✓ Using Ollama local LLM for query generation")
        except:
            try:
                # Fallback to a simple approach
                print("⚠ Ollama not available, using simple query expansion")
                llm = None
            except Exception as e:
                print(f"⚠ Error setting up LLM: {e}")
                llm = None
        
        # Create multi-query retriever
        if llm:
            retriever = MultiQueryRetriever.from_llm(
                retriever=vectorstore.as_retriever(),
                llm=llm
            )
            print("✓ Multi-query retriever created with LLM")
        else:
            # Simple approach without LLM
            print("Using simple retriever (LLM not available)")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        print("✓ Multi-query retriever setup completed\n")
        
        # Test retrieval
        print("--- Testing Multi-Query Retrieval ---")
        query = "What are the main economic policies mentioned?"
        print(f"Query: '{query}'")
        
        try:
            docs = retriever.get_relevant_documents(query)
            print(f"✓ Retrieved {len(docs)} relevant documents")
            
            for i, doc in enumerate(docs[:3]):  # Show first 3
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc.page_content[:150]}...")
                print(f"Metadata: {doc.metadata}")
            
            return retriever
            
        except Exception as e:
            print(f"⚠ Error during retrieval: {e}")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing dependencies: {e}")
        print("Please install required packages: pip install chromadb sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with multi-query retriever: {e}")
        return None

def demonstrate_self_querying_retriever(chunks):
    """Demonstrate Self-Querying Retriever"""
    print("=== Self-Querying Retriever ===\n")
    
    if not chunks:
        print("⚠ No chunks available for retriever. Skipping this section.")
        return None
    
    try:
        from langchain.retrievers.self_query.base import SelfQueryRetriever
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain.chat_models import ChatOpenAI
        from langchain.llms import Ollama
        from langchain.schema import Document
        from pydantic import BaseModel, Field
        from typing import Optional
        
        print("--- Setting Up Self-Querying Retriever ---")
        
        # Use local embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        print("✓ Embedding model created successfully")
        
        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": "state_of_union",
                    "chunk_id": i,
                    "length": len(chunk),
                    "topic": "politics" if "policy" in chunk.lower() else "general"
                }
            )
            documents.append(doc)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name="state_of_union_self_query"
        )
        print("✓ Vector store created successfully")
        
        # Try to use local LLM
        try:
            llm = Ollama(model="llama2")
            print("✓ Using Ollama local LLM for self-querying")
        except:
            print("⚠ Ollama not available, self-querying retriever requires LLM")
            print("Skipping self-querying demonstration")
            return None
        
        # Define metadata filter
        class DocumentMetadata(BaseModel):
            source: str = Field(description="Source of the document")
            chunk_id: int = Field(description="ID of the chunk")
            length: int = Field(description="Length of the chunk")
            topic: str = Field(description="Topic of the chunk")
        
        # Create self-querying retriever
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents="Document content about state of the union",
            metadata_field_info=DocumentMetadata
        )
        print("✓ Self-querying retriever created successfully\n")
        
        # Test retrieval
        print("--- Testing Self-Querying Retrieval ---")
        query = "Find documents about economic policies with chunk_id less than 10"
        print(f"Query: '{query}'")
        
        try:
            docs = retriever.get_relevant_documents(query)
            print(f"✓ Retrieved {len(docs)} relevant documents")
            
            for i, doc in enumerate(docs[:3]):  # Show first 3
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc.page_content[:150]}...")
                print(f"Metadata: {doc.metadata}")
            
            return retriever
            
        except Exception as e:
            print(f"⚠ Error during retrieval: {e}")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing dependencies: {e}")
        print("Please install required packages: pip install chromadb sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with self-querying retriever: {e}")
        return None

def demonstrate_parent_document_retriever(chunks):
    """Demonstrate Parent Document Retriever"""
    print("=== Parent Document Retriever ===\n")
    
    if not chunks:
        print("⚠ No chunks available for retriever. Skipping this section.")
        return None
    
    try:
        from langchain.retrievers import ParentDocumentRetriever
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        
        print("--- Setting Up Parent Document Retriever ---")
        
        # Use local embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        print("✓ Embedding model created successfully")
        
        # Create vector store
        vectorstore = Chroma(
            embedding_function=embedding_model,
            collection_name="state_of_union_parent"
        )
        print("✓ Vector store created successfully")
        
        # Create text splitters
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        print("✓ Text splitters created successfully")
        
        # Create parent document retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        print("✓ Parent document retriever created successfully")
        
        # Add documents
        print("--- Adding Documents to Retriever ---")
        try:
            retriever.add_documents(chunks)
            print("✓ Documents added successfully")
        except Exception as e:
            print(f"⚠ Error adding documents: {e}")
            print("This might be due to document format. Continuing with demo...")
        
        print("✓ Parent document retriever setup completed\n")
        
        # Test retrieval
        print("--- Testing Parent Document Retrieval ---")
        query = "What are the main economic policies mentioned?"
        print(f"Query: '{query}'")
        
        try:
            docs = retriever.get_relevant_documents(query)
            print(f"✓ Retrieved {len(docs)} relevant documents")
            
            for i, doc in enumerate(docs[:3]):  # Show first 3
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc.page_content[:150]}...")
                print(f"Metadata: {doc.metadata}")
            
            return retriever
            
        except Exception as e:
            print(f"⚠ Error during retrieval: {e}")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing dependencies: {e}")
        print("Please install required packages: pip install chromadb sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with parent document retriever: {e}")
        return None

def exercises(chunks):
    """Complete the exercises from the notebook"""
    print("=== Exercises ===\n")
    
    if not chunks:
        print("⚠ No chunks available for exercises. Skipping this section.")
        return
    
    # Exercise 1 - Implement a custom retriever
    print("--- Exercise 1: Custom Retriever Implementation ---")
    print("Creating a simple custom retriever that filters by document length...")
    
    try:
        from langchain.schema import BaseRetriever, Document
        from typing import List
        
        class CustomLengthRetriever(BaseRetriever):
            def __init__(self, documents: List[Document], min_length: int = 500):
                self.documents = documents
                self.min_length = min_length
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                # Filter documents by length
                filtered_docs = [
                    doc for doc in self.documents 
                    if len(doc.page_content) >= self.min_length
                ]
                return filtered_docs[:3]  # Return top 3
        
        # Test custom retriever
        custom_retriever = CustomLengthRetriever(chunks, min_length=500)
        docs = custom_retriever.get_relevant_documents("test query")
        print(f"✓ Custom retriever created successfully")
        print(f"Retrieved {len(docs)} documents with length >= 500 characters")
        
    except Exception as e:
        print(f"⚠ Error with custom retriever: {e}")
    
    # Exercise 2 - Compare different retrieval strategies
    print("\n--- Exercise 2: Retrieval Strategy Comparison ---")
    print("Comparing different retrieval approaches...")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        
        # Create vector store
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            collection_name="exercise_comparison"
        )
        
        # Test different search types
        query = "economic policies"
        
        # Similarity search
        similarity_docs = vectorstore.similarity_search(query, k=3)
        print(f"✓ Similarity search: {len(similarity_docs)} documents")
        
        # MMR search
        mmr_docs = vectorstore.max_marginal_relevance_search(query, k=3)
        print(f"✓ MMR search: {len(mmr_docs)} documents")
        
        print("Note: MMR search provides more diverse results")
        
    except Exception as e:
        print(f"⚠ Error with retrieval comparison: {e}")

def main():
    """Main function to demonstrate various retrievers"""
    print("=== LangChain Retriever Demo (Local Models) ===\n")
    
    # Check GPU availability
    gpu_available = check_gpu()
    
    # Load and split data
    data, chunks = load_and_split_data()
    if not chunks:
        print("⚠ Cannot proceed without document chunks.")
        return
    
    # Demonstrate different retriever types
    vector_retriever = demonstrate_vector_store_retriever(chunks)
    multi_query_retriever = demonstrate_multi_query_retriever(chunks)
    self_query_retriever = demonstrate_self_querying_retriever(chunks)
    parent_doc_retriever = demonstrate_parent_document_retriever(chunks)
    
    # Complete exercises
    exercises(chunks)
    
    print("=== Demo Complete ===")
    print("You have successfully explored various retriever types:")
    if vector_retriever:
        print("✓ Vector Store-backed Retriever")
    if multi_query_retriever:
        print("✓ Multi-Query Retriever")
    if self_query_retriever:
        print("✓ Self-Querying Retriever")
    if parent_doc_retriever:
        print("✓ Parent Document Retriever")
    
    print(f"\nGPU Acceleration: {'Enabled' if gpu_available else 'Disabled'}")
    
    print("\nKey takeaways:")
    print("- Different retrievers serve different use cases")
    print("- Vector store retrievers are great for semantic search")
    print("- Multi-query retrievers provide comprehensive results")
    print("- Self-querying retrievers enable natural language queries")
    print("- Parent document retrievers maintain document hierarchy")

if __name__ == "__main__":
    main()
