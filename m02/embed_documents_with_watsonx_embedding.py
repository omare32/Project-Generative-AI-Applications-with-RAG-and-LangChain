#!/usr/bin/env python3
"""
Embed Documents Using Watsonx's Embedding Model
===============================================

This script demonstrates how to use embedding models from watsonx.ai and Hugging Face 
to embed documents. Document embedding is a powerful technique to convert textual data 
into numerical vectors, which can then be used for various downstream tasks such as 
search, classification, clustering, and more.

Objectives:
- Prepare and preprocess documents for embedding
- Use watsonx.ai and Hugging Face embedding models to generate embeddings for your documents
"""

import warnings
warnings.filterwarnings('ignore')

import os

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
    
    # Load the document using LangChain's TextLoader
    try:
        from langchain_community.document_loaders import TextLoader
        
        loader = TextLoader("state-of-the-union.txt")
        data = loader.load()
        print(f"✓ Document loaded successfully. Number of documents: {len(data)}")
        print(f"Document content preview: {data[0].page_content[:200]}...\n")
        
    except ImportError as e:
        print(f"⚠ Error importing TextLoader: {e}")
        print("Please install langchain-community: pip install langchain-community")
        return None, None
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

def demonstrate_watsonx_embedding(chunks):
    """Demonstrate Watsonx embedding model"""
    print("=== Watsonx Embedding Model ===\n")
    
    if not chunks:
        print("⚠ No chunks available for embedding. Skipping this section.")
        return None
    
    try:
        from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
        from langchain_ibm import WatsonxEmbeddings
        
        print("--- Model Description ---")
        print("Using IBM 'slate-125m-english-rtrvr' model:")
        print("- Standard sentence transformers model based on bi-encoders")
        print("- Produces embeddings for given input (query, passage, document, etc.)")
        print("- Trained to maximize cosine similarity between text pieces")
        print("- ~125 million parameters with embedding dimension of 768")
        print("- Maximum input tokens: 512\n")
        
        # Build the model
        print("--- Building Model ---")
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }
        
        watsonx_embedding = WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="skills-network",
            params=embed_params,
        )
        print("✓ Watsonx embedding model created successfully\n")
        
        # Query embeddings
        print("--- Query Embeddings ---")
        query = "How are you?"
        print(f"Query: '{query}'")
        
        try:
            query_result = watsonx_embedding.embed_query(query)
            print(f"✓ Query embedding generated successfully")
            print(f"Embedding dimension: {len(query_result)}")
            print(f"First 5 values: {query_result[:5]}\n")
        except Exception as e:
            print(f"⚠ Error generating query embedding: {e}")
            print("This may be due to missing IBM Watson AI credentials\n")
            return None
        
        # Document embeddings
        print("--- Document Embeddings ---")
        try:
            doc_result = watsonx_embedding.embed_documents(chunks)
            print(f"✓ Document embeddings generated successfully")
            print(f"Number of document embeddings: {len(doc_result)}")
            print(f"First document embedding dimension: {len(doc_result[0])}")
            print(f"First 5 values of first document: {doc_result[0][:5]}\n")
            
            return watsonx_embedding
            
        except Exception as e:
            print(f"⚠ Error generating document embeddings: {e}")
            print("This may be due to missing IBM Watson AI credentials\n")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing Watsonx dependencies: {e}")
        print("Please install ibm-watsonx-ai and langchain-ibm:")
        print("pip install ibm-watsonx-ai langchain-ibm")
        return None
    except Exception as e:
        print(f"⚠ Error with Watsonx embedding: {e}")
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
        print("Using 'all-mpnet-base-v2' from HuggingFace:")
        print("- Sentence-transformers model")
        print("- Maps sentences and paragraphs to 768-dimensional dense vector space")
        print("- Can be used for clustering or semantic search")
        print("- Pre-trained on Microsoft/money-base and fine-tuned on 1B sentence pairs\n")
        
        # Build the model
        print("--- Building Model ---")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)
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
            print("This may be due to missing model files or network issues\n")
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
            print("This may be due to missing model files or network issues\n")
            return None
            
    except ImportError as e:
        print(f"⚠ Error importing HuggingFace dependencies: {e}")
        print("Please install sentence-transformers:")
        print("pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"⚠ Error with HuggingFace embedding: {e}")
        return None

def exercises(chunks):
    """Complete the exercises from the notebook"""
    print("=== Exercises ===\n")
    
    if not chunks:
        print("⚠ No chunks available for exercises. Skipping this section.")
        return
    
    # Exercise 1 - Using another watsonx embedding model
    print("--- Exercise 1: Using Another Watsonx Embedding Model ---")
    print("Trying to use 'ibm/slate-30m-english-rtrvr' model...")
    
    try:
        from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
        from langchain_ibm import WatsonxEmbeddings
        
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }
        
        watsonx_embedding_30m = WatsonxEmbeddings(
            model_id="ibm/slate-30m-english-rtrvr",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="skills-network",
            params=embed_params,
        )
        
        print("✓ Alternative Watsonx embedding model created successfully")
        
        try:
            doc_result = watsonx_embedding_30m.embed_documents(chunks)
            print(f"✓ Document embeddings generated with alternative model")
            print(f"First 5 values of first document: {doc_result[0][:5]}")
            print("Note: This model has fewer parameters (30M vs 125M) but similar functionality\n")
        except Exception as e:
            print(f"⚠ Error generating embeddings with alternative model: {e}")
            print("This may be due to missing IBM Watson AI credentials\n")
            
    except ImportError as e:
        print(f"⚠ Error importing dependencies for exercise: {e}")
    except Exception as e:
        print(f"⚠ Error with exercise: {e}")

def main():
    """Main function to demonstrate document embedding"""
    print("=== Document Embedding Demo ===\n")
    
    # Load and split data
    data, chunks = load_and_split_data()
    if not chunks:
        print("⚠ Cannot proceed without document chunks.")
        return
    
    # Demonstrate Watsonx embedding
    watsonx_model = demonstrate_watsonx_embedding(chunks)
    
    # Demonstrate HuggingFace embedding
    huggingface_model = demonstrate_huggingface_embedding(chunks)
    
    # Complete exercises
    exercises(chunks)
    
    print("=== Demo Complete ===")
    print("You have successfully explored document embedding using:")
    if watsonx_model:
        print("✓ Watsonx.ai embedding models")
    if huggingface_model:
        print("✓ HuggingFace embedding models")
    
    print("\nKey takeaways:")
    print("- Document embedding converts text to numerical vectors")
    print("- These vectors capture semantic meaning for downstream tasks")
    print("- Different models may produce different embedding dimensions")
    print("- Embeddings enable similarity search and semantic understanding")

if __name__ == "__main__":
    main()
