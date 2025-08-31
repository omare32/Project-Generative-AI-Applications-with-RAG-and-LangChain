#!/usr/bin/env python3
"""
Embed Documents Using Local Models (GPU Optimized)
=================================================

This script demonstrates how to use local embedding models optimized for GPU 
to embed documents. Since you have a 4090 GPU, we'll use models that can 
leverage GPU acceleration for better performance.

Objectives:
- Prepare and preprocess documents for embedding
- Use local embedding models with GPU acceleration
- Generate embeddings for queries and documents
- Compare different local embedding models
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
            chunk_size=100,
            chunk_overlap=20,
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

def demonstrate_sentence_transformers_embedding(chunks):
    """Demonstrate Sentence Transformers embedding model"""
    print("=== Sentence Transformers Embedding Model ===\n")
    
    if not chunks:
        print("⚠ No chunks available for embedding. Skipping this section.")
        return None
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("--- Model Description ---")
        print("Using 'all-mpnet-base-v2' from Sentence Transformers:")
        print("- Maps sentences and paragraphs to 768-dimensional dense vector space")
        print("- Can be used for clustering or semantic search")
        print("- Pre-trained on Microsoft/money-base and fine-tuned on 1B sentence pairs")
        print("- Optimized for GPU acceleration\n")
        
        # Build the model
        print("--- Building Model ---")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = SentenceTransformer(model_name, device=device)
        print("✓ Sentence Transformers model created successfully\n")
        
        # Query embeddings
        print("--- Query Embeddings ---")
        query = "How are you?"
        print(f"Query: '{query}'")
        
        try:
            query_result = model.encode(query)
            print(f"✓ Query embedding generated successfully")
            print(f"Embedding dimension: {len(query_result)}")
            print(f"First 5 values: {query_result[:5]}\n")
        except Exception as e:
            print(f"⚠ Error generating query embedding: {e}")
            return None
        
        # Document embeddings
        print("--- Document Embeddings ---")
        try:
            doc_result = model.encode(chunks)
            print(f"✓ Document embeddings generated successfully")
            print(f"Number of document embeddings: {len(doc_result)}")
            print(f"First document embedding dimension: {len(doc_result[0])}")
            print(f"First 5 values of first document: {doc_result[0][:5]}\n")
            
            return model
            
        except Exception as e:
            print(f"⚠ Error generating document embeddings: {e}")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing SentenceTransformers: {e}")
        print("Please install sentence-transformers: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with SentenceTransformers embedding: {e}")
        return None

def demonstrate_huggingface_embedding(chunks):
    """Demonstrate Hugging Face embedding model"""
    print("=== Hugging Face Embedding Model ===\n")
    
    if not chunks:
        print("⚠ No chunks available for embedding. Skipping this section.")
        return None
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        print("--- Model Description ---")
        print("Using 'all-MiniLM-L6-v2' from HuggingFace:")
        print("- Lightweight model with 384-dimensional embeddings")
        print("- Fast inference and good quality")
        print("- Optimized for speed and efficiency\n")
        
        # Build the model
        print("--- Building Model ---")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        huggingface_embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        print("✓ HuggingFace embedding model created successfully\n")
        
        # Query embeddings
        print("--- Query Embeddings ---")
        query = "How are you?"
        print(f"Query: '{query}'")
        
        try:
            query_result = huggingface_embedding.embed_query(query)
            print(f"✓ Query embedding generated successfully")
            print(f"Embedding dimension: {len(query_result)}")
            print(f"First 5 values: {query_result[:5]}\n")
        except Exception as e:
            print(f"⚠ Error generating query embedding: {e}")
            return None
        
        # Document embeddings
        print("--- Document Embeddings ---")
        try:
            doc_result = huggingface_embedding.embed_documents(chunks)
            print(f"✓ Document embeddings generated successfully")
            print(f"Number of document embeddings: {len(doc_result)}")
            print(f"First document embedding dimension: {len(doc_result[0])}")
            print(f"First 5 values of first document: {doc_result[0][:5]}\n")
            
            return huggingface_embedding
            
        except Exception as e:
            print(f"⚠ Error generating document embeddings: {e}")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing HuggingFace dependencies: {e}")
        print("Please install sentence-transformers: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with HuggingFace embedding: {e}")
        return None

def demonstrate_custom_embedding(chunks):
    """Demonstrate custom embedding using transformers directly"""
    print("=== Custom Transformers Embedding ===\n")
    
    if not chunks:
        print("⚠ No chunks available for embedding. Skipping this section.")
        return None
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch.nn.functional as F
        
        print("--- Model Description ---")
        print("Using 'sentence-transformers/all-mpnet-base-v2' with custom implementation:")
        print("- Direct use of transformers library")
        print("- More control over the embedding process")
        print("- Can be customized for specific use cases\n")
        
        # Build the model
        print("--- Building Model ---")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        print("✓ Custom transformers model created successfully\n")
        
        # Query embeddings
        print("--- Query Embeddings ---")
        query = "How are you?"
        print(f"Query: '{query}'")
        
        try:
            # Tokenize and encode
            inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                query_result = embeddings.cpu().numpy()[0]
            
            print(f"✓ Query embedding generated successfully")
            print(f"Embedding dimension: {len(query_result)}")
            print(f"First 5 values: {query_result[:5]}\n")
        except Exception as e:
            print(f"⚠ Error generating query embedding: {e}")
            return None
        
        # Document embeddings
        print("--- Document Embeddings ---")
        try:
            doc_results = []
            for chunk in chunks[:5]:  # Process first 5 chunks for demo
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    doc_results.append(embeddings.cpu().numpy()[0])
            
            print(f"✓ Document embeddings generated successfully")
            print(f"Number of document embeddings: {len(doc_results)}")
            print(f"First document embedding dimension: {len(doc_results[0])}")
            print(f"First 5 values of first document: {doc_results[0][:5]}\n")
            
            return {"tokenizer": tokenizer, "model": model}
            
        except Exception as e:
            print(f"⚠ Error generating document embeddings: {e}")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing transformers: {e}")
        print("Please install transformers: pip install transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with custom embedding: {e}")
        return None

def exercises(chunks):
    """Complete the exercises from the notebook"""
    print("=== Exercises ===\n")
    
    if not chunks:
        print("⚠ No chunks available for exercises. Skipping this section.")
        return
    
    # Exercise 1 - Using another embedding model
    print("--- Exercise 1: Using Another Embedding Model ---")
    print("Trying to use 'sentence-transformers/all-MiniLM-L6-v2' model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name, device=device)
        
        print("✓ Alternative embedding model created successfully")
        
        try:
            doc_result = model.encode(chunks[:3])  # Process first 3 chunks
            print(f"✓ Document embeddings generated with alternative model")
            print(f"First 5 values of first document: {doc_result[0][:5]}")
            print("Note: This model has smaller embeddings (384 vs 768) but is faster\n")
        except Exception as e:
            print(f"⚠ Error generating embeddings with alternative model: {e}\n")
            
    except ImportError as e:
        print(f"⚠ Error importing dependencies for exercise: {e}")
    except Exception as e:
        print(f"⚠ Error with exercise: {e}")

def main():
    """Main function to demonstrate local document embedding"""
    print("=== Local Document Embedding Demo (GPU Optimized) ===\n")
    
    # Check GPU availability
    gpu_available = check_gpu()
    
    # Load and split data
    data, chunks = load_and_split_data()
    if not chunks:
        print("⚠ Cannot proceed without document chunks.")
        return
    
    # Demonstrate different embedding approaches
    sentence_transformers_model = demonstrate_sentence_transformers_embedding(chunks)
    huggingface_model = demonstrate_huggingface_embedding(chunks)
    custom_model = demonstrate_custom_embedding(chunks)
    
    # Complete exercises
    exercises(chunks)
    
    print("=== Demo Complete ===")
    print("You have successfully explored local document embedding using:")
    if sentence_transformers_model:
        print("✓ Sentence Transformers (all-mpnet-base-v2)")
    if huggingface_model:
        print("✓ HuggingFace Embeddings (all-MiniLM-L6-v2)")
    if custom_model:
        print("✓ Custom Transformers implementation")
    
    print(f"\nGPU Acceleration: {'Enabled' if gpu_available else 'Disabled'}")
    
    print("\nKey takeaways:")
    print("- Local models provide privacy and control")
    print("- GPU acceleration significantly improves performance")
    print("- Different models offer different trade-offs (speed vs quality)")
    print("- Custom implementations allow for specific optimizations")

if __name__ == "__main__":
    main()
