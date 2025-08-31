#!/usr/bin/env python3
"""
Create and Configure a Vector Database to Store Document Embeddings
==================================================================

This script demonstrates how to use vector databases to store embeddings generated from 
textual data using LangChain. The focus will be on two popular vector databases: 
Chroma DB and FAISS (Facebook AI Similarity Search). You will also learn how to perform 
similarity searches in these databases based on a query, enabling efficient retrieval 
of relevant information.

Objectives:
- Prepare and preprocess documents for embeddings
- Generate embeddings using watsonx.ai's embedding model
- Store these embeddings in Chroma DB and FAISS
- Perform similarity searches to retrieve relevant documents based on new inquiries
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

def load_and_split_text():
    """Load and split the source document"""
    print("=== Loading and Splitting Text ===\n")
    
    # Download the source document
    if not download_file(
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/BYlUHaillwM8EUItaIytHQ/companypolicies.txt",
        "companypolicies.txt"
    ):
        print("⚠ Cannot proceed without the sample document.")
        return None, None
    
    # Load the document using LangChain's TextLoader
    try:
        from langchain_community.document_loaders import TextLoader
        
        loader = TextLoader("companypolicies.txt")
        data = loader.load()
        print(f"✓ Document loaded successfully. Number of documents: {len(data)}")
        print(f"Document preview: {data[0].page_content[:200]}...\n")
        
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
        
        chunks = text_splitter.split_documents(data)
        print(f"✓ Document split into {len(chunks)} chunks")
        print(f"First chunk preview: {chunks[0].page_content[:100]}...\n")
        
        return data, chunks
        
    except ImportError as e:
        print(f"⚠ Error importing RecursiveCharacterTextSplitter: {e}")
        print("Please install langchain: pip install langchain")
        return data, None
    except Exception as e:
        print(f"⚠ Error splitting document: {e}")
        return data, None

def build_embedding_model():
    """Build the embedding model using watsonx.ai"""
    print("=== Building Embedding Model ===\n")
    
    try:
        from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
        from langchain_ibm import WatsonxEmbeddings
        
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
        
        print("✓ Watsonx embedding model created successfully")
        print("Model: ibm/slate-125m-english-rtrvr")
        print("Embedding dimension: 768\n")
        
        return watsonx_embedding
        
    except ImportError as e:
        print(f"⚠ Error importing Watsonx dependencies: {e}")
        print("Please install ibm-watsonx-ai and langchain-ibm:")
        print("pip install ibm-watsonx-ai langchain-ibm")
        return None
    except Exception as e:
        print(f"⚠ Error building embedding model: {e}")
        return None

def demonstrate_chroma_db(chunks, embedding_model):
    """Demonstrate Chroma DB vector store"""
    print("=== Chroma DB Vector Store ===\n")
    
    if not chunks or not embedding_model:
        print("⚠ No chunks or embedding model available. Skipping this section.")
        return None
    
    try:
        from langchain.vectorstores import Chroma
        
        print("--- Building the Database ---")
        
        # Create ID list for unique identifiers
        ids = [str(i) for i in range(0, len(chunks))]
        print(f"✓ Created {len(ids)} unique IDs for chunks")
        
        # Create embeddings and store in Chroma database
        vectordb = Chroma.from_documents(chunks, embedding_model, ids=ids)
        print("✓ Vector database created successfully")
        
        # Display some chunks from the database
        print("\n--- Displaying Chunks from Database ---")
        for i in range(3):
            result = vectordb._collection.get(ids=str(i))
            print(f"Chunk {i}: {result['documents'][0][:100]}...")
        
        # Check database count
        count = vectordb._collection.count()
        print(f"\n✓ Total chunks in database: {count}")
        
        return vectordb
        
    except ImportError as e:
        print(f"⚠ Error importing Chroma: {e}")
        print("Please install chromadb: pip install chromadb")
        return None
    except Exception as e:
        print(f"⚠ Error with Chroma DB: {e}")
        return None

def demonstrate_chroma_similarity_search(vectordb):
    """Demonstrate similarity search in Chroma DB"""
    print("\n--- Similarity Search in Chroma DB ---")
    
    if not vectordb:
        print("⚠ No vector database available. Skipping this section.")
        return
    
    try:
        query = "Email policy"
        print(f"Query: '{query}'")
        
        # Default similarity search (top 4 results)
        docs = vectordb.similarity_search(query)
        print(f"✓ Default search retrieved {len(docs)} documents")
        print("First result preview:")
        print(f"  {docs[0].page_content[:200]}...\n")
        
        # Search with k=1 (top 1 result)
        print("--- Search with k=1 ---")
        single_doc = vectordb.similarity_search(query, k=1)
        print(f"✓ Retrieved top 1 result:")
        print(f"  {single_doc[0].page_content[:200]}...\n")
        
    except Exception as e:
        print(f"⚠ Error with similarity search: {e}")

def demonstrate_faiss_db(chunks, embedding_model):
    """Demonstrate FAISS vector store"""
    print("\n=== FAISS Vector Store ===\n")
    
    if not chunks or not embedding_model:
        print("⚠ No chunks or embedding model available. Skipping this section.")
        return None
    
    try:
        from langchain_community.vectorstores import FAISS
        
        print("--- Building FAISS Database ---")
        
        # Create ID list for unique identifiers
        ids = [str(i) for i in range(0, len(chunks))]
        
        # Create embeddings and store in FAISS database
        faissdb = FAISS.from_documents(chunks, embedding_model, ids=ids)
        print("✓ FAISS vector database created successfully")
        
        # Display some chunks from the database
        print("\n--- Displaying Chunks from FAISS Database ---")
        for i in range(3):
            result = faissdb.docstore.search(str(i))
            print(f"Chunk {i}: {result.page_content[:100]}...")
        
        return faissdb
        
    except ImportError as e:
        print(f"⚠ Error importing FAISS: {e}")
        print("Please install faiss-cpu: pip install faiss-cpu")
        return None
    except Exception as e:
        print(f"⚠ Error with FAISS DB: {e}")
        return None

def demonstrate_faiss_similarity_search(faissdb):
    """Demonstrate similarity search in FAISS DB"""
    print("\n--- Similarity Search in FAISS DB ---")
    
    if not faissdb:
        print("⚠ No FAISS database available. Skipping this section.")
        return
    
    try:
        query = "Email policy"
        print(f"Query: '{query}'")
        
        docs = faissdb.similarity_search(query)
        print(f"✓ FAISS search retrieved {len(docs)} documents")
        print("First result preview:")
        print(f"  {docs[0].page_content[:200]}...\n")
        
        print("Note: Results should be similar to Chroma DB as both use the same embedding model")
        
    except Exception as e:
        print(f"⚠ Error with FAISS similarity search: {e}")

def demonstrate_vector_store_management(vectordb):
    """Demonstrate managing vector store: adding, updating, and deleting entries"""
    print("\n=== Managing Vector Store ===\n")
    
    if not vectordb:
        print("⚠ No vector database available. Skipping this section.")
        return
    
    try:
        from langchain_core.documents import Document
        
        print("--- Adding New Document ---")
        
        # Create new text
        text = "Instructlab is the best open source tool for fine-tuning a LLM."
        
        # Form text into Document object
        new_chunk = Document(
            page_content=text,
            metadata={
                "source": "ibm.com",
                "page": 1
            }
        )
        
        new_chunks = [new_chunk]
        print(f"✓ Created new document: '{text}'")
        
        # Check if ID 215 exists (should be empty)
        print("\n--- Checking Existing ID 215 ---")
        existing = vectordb._collection.get(ids=['215'])
        print(f"ID 215 content: {existing}")
        
        # Add new document
        print("\n--- Adding Document to Database ---")
        vectordb.add_documents(new_chunks, ids=["215"])
        print("✓ New document added with ID 215")
        
        # Check count
        count = vectordb._collection.count()
        print(f"✓ Total chunks in database: {count}")
        
        # Display newly added document
        print("\n--- Displaying Newly Added Document ---")
        new_doc = vectordb._collection.get(ids=['215'])
        print(f"New document: {new_doc['documents'][0]}")
        
        # Update document
        print("\n--- Updating Document ---")
        update_chunk = Document(
            page_content="Instructlab is a perfect open source tool for fine-tuning a LLM.",
            metadata={
                "source": "ibm.com",
                "page": 1
            }
        )
        
        vectordb.update_document('215', update_chunk)
        print("✓ Document updated successfully")
        
        # Display updated document
        updated_doc = vectordb._collection.get(ids=['215'])
        print(f"Updated document: {updated_doc['documents'][0]}")
        
        # Delete document
        print("\n--- Deleting Document ---")
        vectordb._collection.delete(ids=['215'])
        print("✓ Document deleted successfully")
        
        # Verify deletion
        deleted_doc = vectordb._collection.get(ids=['215'])
        print(f"Deleted document content: {deleted_doc}")
        
    except Exception as e:
        print(f"⚠ Error with vector store management: {e}")

def exercises(vectordb):
    """Complete the exercises from the notebook"""
    print("\n=== Exercises ===\n")
    
    if not vectordb:
        print("⚠ No vector database available for exercises. Skipping this section.")
        return
    
    # Exercise 1: Use another query to conduct similarity search
    print("--- Exercise 1: Use Another Query for Similarity Search ---")
    print("Query: 'Smoking policy'")
    
    try:
        docs = vectordb.similarity_search("Smoking policy")
        print(f"✓ Retrieved {len(docs)} documents for 'Smoking policy'")
        print("First result preview:")
        print(f"  {docs[0].page_content[:200]}...\n")
        
    except Exception as e:
        print(f"⚠ Error with exercise: {e}")

def main():
    """Main function to demonstrate vector stores"""
    print("=== LangChain Vector Store Demo ===\n")
    
    # Load and split text
    data, chunks = load_and_split_text()
    if not chunks:
        print("⚠ Cannot proceed without document chunks.")
        return
    
    # Build embedding model
    embedding_model = build_embedding_model()
    if not embedding_model:
        print("⚠ Cannot proceed without embedding model.")
        return
    
    # Demonstrate Chroma DB
    vectordb = demonstrate_chroma_db(chunks, embedding_model)
    if vectordb:
        demonstrate_chroma_similarity_search(vectordb)
    
    # Demonstrate FAISS DB
    faissdb = demonstrate_faiss_db(chunks, embedding_model)
    if faissdb:
        demonstrate_faiss_similarity_search(faissdb)
    
    # Demonstrate vector store management
    if vectordb:
        demonstrate_vector_store_management(vectordb)
    
    # Complete exercises
    if vectordb:
        exercises(vectordb)
    
    print("\n=== Demo Complete ===")
    print("You have successfully explored vector databases using LangChain!")
    print("\nKey takeaways:")
    print("- Chroma DB: Easy-to-use vector database with good performance")
    print("- FAISS: High-performance vector similarity search library")
    print("- Both databases can store and retrieve document embeddings efficiently")
    print("- Vector stores support CRUD operations (Create, Read, Update, Delete)")
    print("- Similarity search enables semantic document retrieval")

if __name__ == "__main__":
    main()
