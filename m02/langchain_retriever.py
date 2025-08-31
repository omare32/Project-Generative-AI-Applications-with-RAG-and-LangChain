#!/usr/bin/env python3
"""
Develop a Retriever to Fetch Document Segments Based on Queries
==============================================================

This script demonstrates how to use various retrievers to efficiently extract relevant 
document segments from text using LangChain. You will learn about four types of retrievers: 
Vector Store-backed Retriever, Multi-Query Retriever, Self-Querying Retriever, and 
Parent Document Retriever.

Objectives:
- Use various types of retrievers to efficiently extract relevant document segments from text
- Apply the Vector Store-backed Retriever for semantic similarity and relevance
- Utilize the Multi-Query Retriever for comprehensive results
- Implement the Self-Querying Retriever for automatic query refinement
- Employ the Parent Document Retriever to maintain context and relevance
"""

import warnings
warnings.warn = lambda *args, **kwargs: None
warnings.filterwarnings('ignore')

import os
import logging

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

def build_llm():
    """Build LLM using IBM Watson AI"""
    print("--- Building LLM ---")
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
        
        model_id = 'mistralai/mixtral-8x7b-instruct-v01'
        
        parameters = {
            GenParams.MAX_NEW_TOKENS: 256,
            GenParams.TEMPERATURE: 0.5,
        }
        
        credentials = {
            "url": "https://us-south.ml.cloud.ibm.com"
        }
        
        project_id = "skills-network"
        
        model = ModelInference(
            model_id=model_id,
            params=parameters,
            credentials=credentials,
            project_id=project_id
        )
        
        mixtral_llm = WatsonxLLM(model=model)
        print("✓ LLM built successfully")
        return mixtral_llm
        
    except ImportError as e:
        print(f"⚠ Error importing LLM dependencies: {e}")
        print("Please install ibm-watsonx-ai")
        return None
    except Exception as e:
        print(f"⚠ Error building LLM: {e}")
        return None

def text_splitter(data, chunk_size, chunk_overlap):
    """Split documents into chunks"""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(data)
        print(f"✓ Text split into {len(chunks)} chunks")
        return chunks
        
    except ImportError as e:
        print(f"⚠ Error importing text splitter: {e}")
        print("Please install langchain")
        return None
    except Exception as e:
        print(f"⚠ Error splitting text: {e}")
        return None

def watsonx_embedding():
    """Build Watsonx embedding model"""
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
        print("✓ Watsonx embedding model built successfully")
        return watsonx_embedding
        
    except ImportError as e:
        print(f"⚠ Error importing embedding dependencies: {e}")
        print("Please install ibm-watsonx-ai and langchain-ibm")
        return None
    except Exception as e:
        print(f"⚠ Error building embedding model: {e}")
        return None

def demonstrate_vector_store_retriever():
    """Demonstrate Vector Store-Backed Retriever"""
    print("\n=== Vector Store-Backed Retriever ===\n")
    
    # Download and load company policies document
    if not download_file(
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/MZ9z1lm-Ui3YBp3SYWLTAQ/companypolicies.txt",
        "companypolicies.txt"
    ):
        print("⚠ Cannot proceed without the company policies document.")
        return None
    
    try:
        from langchain_community.document_loaders import TextLoader
        
        loader = TextLoader("companypolicies.txt")
        txt_data = loader.load()
        print(f"✓ Company policies document loaded. Number of documents: {len(txt_data)}")
        print(f"Document preview: {txt_data[0].page_content[:200]}...\n")
        
    except ImportError as e:
        print(f"⚠ Error importing TextLoader: {e}")
        return None
    except Exception as e:
        print(f"⚠ Error loading document: {e}")
        return None
    
    # Split document into chunks
    chunks_txt = text_splitter(txt_data, 200, 20)
    if not chunks_txt:
        return None
    
    # Build embedding model
    embedding_model = watsonx_embedding()
    if not embedding_model:
        return None
    
    # Store embeddings in ChromaDB
    try:
        from langchain.vectorstores import Chroma
        
        vectordb = Chroma.from_documents(chunks_txt, embedding_model)
        print("✓ Vector database created successfully\n")
        
    except ImportError as e:
        print(f"⚠ Error importing Chroma: {e}")
        print("Please install chromadb")
        return None
    except Exception as e:
        print(f"⚠ Error creating vector database: {e}")
        return None
    
    # Simple similarity search
    print("--- Simple Similarity Search ---")
    query = "email policy"
    retriever = vectordb.as_retriever()
    
    try:
        docs = retriever.invoke(query)
        print(f"✓ Retrieved {len(docs)} documents for query: '{query}'")
        print("First document preview:")
        print(f"  {docs[0].page_content[:200]}...\n")
        
    except Exception as e:
        print(f"⚠ Error with similarity search: {e}")
    
    # MMR retrieval
    print("--- MMR Retrieval ---")
    try:
        retriever = vectordb.as_retriever(search_type="mmr")
        docs = retriever.invoke(query)
        print(f"✓ MMR retrieval completed. Retrieved {len(docs)} documents\n")
        
    except Exception as e:
        print(f"⚠ Error with MMR retrieval: {e}")
    
    # Similarity score threshold retrieval
    print("--- Similarity Score Threshold Retrieval ---")
    try:
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.4}
        )
        docs = retriever.invoke(query)
        print(f"✓ Threshold retrieval completed. Retrieved {len(docs)} documents\n")
        
    except Exception as e:
        print(f"⚠ Error with threshold retrieval: {e}")
    
    return vectordb, chunks_txt

def demonstrate_multi_query_retriever(vectordb):
    """Demonstrate Multi-Query Retriever"""
    print("\n=== Multi-Query Retriever ===\n")
    
    if not vectordb:
        print("⚠ No vector database available. Skipping this section.")
        return
    
    # Download PDF document
    if not download_file(
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf",
        "langchain-paper.pdf"
    ):
        print("⚠ Cannot proceed without the PDF document.")
        return
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        
        loader = PyPDFLoader("langchain-paper.pdf")
        pdf_data = loader.load()
        print(f"✓ LangChain paper loaded. Number of pages: {len(pdf_data)}")
        print(f"First page preview: {pdf_data[0].page_content[:200]}...\n")
        
    except ImportError as e:
        print(f"⚠ Error importing PyPDFLoader: {e}")
        print("Please install pypdf")
        return
    except Exception as e:
        print(f"⚠ Error loading PDF: {e}")
        return
    
    # Split PDF and update vector database
    chunks_pdf = text_splitter(pdf_data, 500, 20)
    if not chunks_pdf:
        return
    
    try:
        # Clear existing embeddings and add new ones
        ids = vectordb.get()["ids"]
        vectordb.delete(ids)
        vectordb = Chroma.from_documents(documents=chunks_pdf, embedding=watsonx_embedding())
        print("✓ Vector database updated with PDF content\n")
        
    except Exception as e:
        print(f"⚠ Error updating vector database: {e}")
        return
    
    # Multi-Query Retriever
    print("--- Multi-Query Retriever ---")
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
        
        query = "What does the paper say about langchain?"
        
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectordb.as_retriever(), 
            llm=build_llm()
        )
        
        # Set logging for queries
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        
        docs = retriever.invoke(query)
        print(f"✓ Multi-query retrieval completed. Retrieved {len(docs)} documents")
        print("Note: Check the logs above to see the generated queries\n")
        
    except ImportError as e:
        print(f"⚠ Error importing MultiQueryRetriever: {e}")
    except Exception as e:
        print(f"⚠ Error with multi-query retrieval: {e}")

def demonstrate_self_querying_retriever():
    """Demonstrate Self-Querying Retriever"""
    print("\n=== Self-Querying Retriever ===\n")
    
    try:
        from langchain_core.documents import Document
        from langchain.chains.query_constructor.base import AttributeInfo
        from langchain.retrievers.self_query.base import SelfQueryRetriever
        
        # Create sample documents with metadata
        docs = [
            Document(
                page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
                metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
            ),
            Document(
                page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
                metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
            ),
            Document(
                page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
                metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
            ),
            Document(
                page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
                metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
            ),
            Document(
                page_content="Toys come alive and have a blast doing so",
                metadata={"year": 1995, "genre": "animated"},
            ),
            Document(
                page_content="Three men walk into the Zone, three men walk out of the Zone",
                metadata={
                    "year": 1979,
                    "director": "Andrei Tarkovsky",
                    "genre": "thriller",
                    "rating": 9.9,
                },
            ),
        ]
        
        print(f"✓ Created {len(docs)} sample movie documents with metadata\n")
        
    except ImportError as e:
        print(f"⚠ Error importing Document dependencies: {e}")
        print("Please install langchain-core")
        return
    except Exception as e:
        print(f"⚠ Error creating sample documents: {e}")
        return
    
    # Define metadata field information
    metadata_field_info = [
        AttributeInfo(
            name="genre",
            description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(
            name="rating", 
            description="A 1-10 rating for the movie", 
            type="float"
        ),
    ]
    
    # Build embedding model and vector database
    embedding_model = watsonx_embedding()
    if not embedding_model:
        return
    
    try:
        from langchain.vectorstores import Chroma
        
        vectordb = Chroma.from_documents(docs, embedding_model)
        print("✓ Vector database created for movie documents\n")
        
    except Exception as e:
        print(f"⚠ Error creating vector database: {e}")
        return
    
    # Self-Querying Retriever
    print("--- Self-Querying Retriever ---")
    try:
        llm_model = build_llm()
        if not llm_model:
            print("⚠ Cannot proceed without LLM model.")
            return
        
        document_content_description = "Brief summary of a movie."
        
        retriever = SelfQueryRetriever.from_llm(
            llm_model,
            vectordb,
            document_content_description,
            metadata_field_info,
        )
        
        print("✓ Self-querying retriever created successfully\n")
        
        # Test queries
        print("--- Testing Self-Querying Retriever ---")
        
        # Filter only
        print("Query: 'I want to watch a movie rated higher than 8.5'")
        try:
            result = retriever.invoke("I want to watch a movie rated higher than 8.5")
            print(f"✓ Retrieved {len(result)} documents\n")
        except Exception as e:
            print(f"⚠ Error with filter query: {e}\n")
        
        # Query with filter
        print("Query: 'Has Greta Gerwig directed any movies about women'")
        try:
            result = retriever.invoke("Has Greta Gerwig directed any movies about women")
            print(f"✓ Retrieved {len(result)} documents\n")
        except Exception as e:
            print(f"⚠ Error with query + filter: {e}\n")
        
        # Composite filter
        print("Query: 'What is a highly rated (above 8.5) science fiction film?'")
        try:
            result = retriever.invoke("What's a highly rated (above 8.5) science fiction film?")
            print(f"✓ Retrieved {len(result)} documents\n")
        except Exception as e:
            print(f"⚠ Error with composite filter: {e}\n")
        
    except ImportError as e:
        print(f"⚠ Error importing SelfQueryRetriever: {e}")
        print("Please install langchain and lark")
    except Exception as e:
        print(f"⚠ Error with self-querying retriever: {e}")

def demonstrate_parent_document_retriever(chunks_txt):
    """Demonstrate Parent Document Retriever"""
    print("\n=== Parent Document Retriever ===\n")
    
    if not chunks_txt:
        print("⚠ No text chunks available. Skipping this section.")
        return
    
    try:
        from langchain.retrievers import ParentDocumentRetriever
        from langchain_text_splitters import CharacterTextSplitter
        from langchain.storage import InMemoryStore
        
        # Set two splitters: parent (big chunks) and child (small chunks)
        parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator='\n')
        child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator='\n')
        
        print("✓ Text splitters configured")
        print(f"  Parent chunk size: 2000, Child chunk size: 400\n")
        
    except ImportError as e:
        print(f"⚠ Error importing ParentDocumentRetriever dependencies: {e}")
        print("Please install langchain and langchain-text-splitters")
        return
    except Exception as e:
        print(f"⚠ Error configuring text splitters: {e}")
        return
    
    # Build embedding model
    embedding_model = watsonx_embedding()
    if not embedding_model:
        return
    
    try:
        from langchain.vectorstores import Chroma
        
        vectordb = Chroma(
            collection_name="split_parents", 
            embedding_function=embedding_model
        )
        
        # Storage layer for parent documents
        store = InMemoryStore()
        
        retriever = ParentDocumentRetriever(
            vectorstore=vectordb,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        print("✓ Parent document retriever created successfully\n")
        
    except Exception as e:
        print(f"⚠ Error creating parent document retriever: {e}")
        return
    
    # Add documents
    try:
        retriever.add_documents(chunks_txt)
        print("✓ Documents added to retriever")
        print(f"Number of large chunks: {len(list(store.yield_keys()))}\n")
        
    except Exception as e:
        print(f"⚠ Error adding documents: {e}")
        return
    
    # Test retrieval
    print("--- Testing Parent Document Retriever ---")
    try:
        # First, check underlying vector store retrieves small chunks
        sub_docs = vectordb.similarity_search("smoking policy")
        print("Small chunk retrieved:")
        print(f"  {sub_docs[0].page_content[:200]}...\n")
        
        # Then, retrieve the relevant large chunk
        retrieved_docs = retriever.invoke("smoking policy")
        print("Large chunk retrieved:")
        print(f"  {retrieved_docs[0].page_content[:200]}...\n")
        
    except Exception as e:
        print(f"⚠ Error testing retrieval: {e}")

def exercises(chunks_txt):
    """Complete the exercises from the notebook"""
    print("\n=== Exercises ===\n")
    
    if not chunks_txt:
        print("⚠ No text chunks available for exercises. Skipping this section.")
        return
    
    # Exercise 1: Retrieve top 2 results using vector store-backed retriever
    print("--- Exercise 1: Retrieve Top 2 Results ---")
    print("Query: 'smoking policy'")
    
    try:
        embedding_model = watsonx_embedding()
        if not embedding_model:
            return
        
        from langchain.vectorstores import Chroma
        
        vectordb = Chroma.from_documents(documents=chunks_txt, embedding=embedding_model)
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        
        docs = retriever.invoke("smoking policy")
        print(f"✓ Retrieved top 2 results: {len(docs)} documents")
        print("First result preview:")
        print(f"  {docs[0].page_content[:200]}...\n")
        
    except Exception as e:
        print(f"⚠ Error with exercise 1: {e}\n")
    
    # Exercise 2: Self-Querying Retriever for a query
    print("--- Exercise 2: Self-Querying Retriever ---")
    print("Query: 'I want to watch a movie directed by Christopher Nolan'")
    
    try:
        # Create sample movie documents
        from langchain_core.documents import Document
        from langchain.chains.query_constructor.base import AttributeInfo
        from langchain.retrievers.self_query.base import SelfQueryRetriever
        
        movie_docs = [
            Document(
                page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
                metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
            ),
            Document(
                page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
                metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
            ),
        ]
        
        metadata_field_info = [
            AttributeInfo(
                name="director",
                description="The name of the movie director",
                type="string",
            ),
        ]
        
        vectordb = Chroma.from_documents(movie_docs, embedding_model)
        
        llm_model = build_llm()
        if llm_model:
            retriever = SelfQueryRetriever.from_llm(
                llm_model,
                vectordb,
                "Brief summary of a movie.",
                metadata_field_info,
            )
            
            result = retriever.invoke("I want to watch a movie directed by Christopher Nolan")
            print(f"✓ Retrieved {len(result)} documents")
            print("Result preview:")
            print(f"  {result[0].page_content[:200]}...\n")
        else:
            print("⚠ Cannot complete exercise without LLM model\n")
            
    except Exception as e:
        print(f"⚠ Error with exercise 2: {e}\n")

def main():
    """Main function to demonstrate all retrievers"""
    print("=== LangChain Retriever Demo ===\n")
    
    # Build helper functions
    print("=== Building Helper Functions ===\n")
    
    llm_model = build_llm()
    if not llm_model:
        print("⚠ Warning: LLM not available. Some retrievers may not work properly.")
    
    embedding_model = watsonx_embedding()
    if not embedding_model:
        print("⚠ Error: Cannot proceed without embedding model.")
        return
    
    # Demonstrate Vector Store-Backed Retriever
    vectordb, chunks_txt = demonstrate_vector_store_retriever()
    
    # Demonstrate Multi-Query Retriever
    demonstrate_multi_query_retriever(vectordb)
    
    # Demonstrate Self-Querying Retriever
    demonstrate_self_querying_retriever()
    
    # Demonstrate Parent Document Retriever
    demonstrate_parent_document_retriever(chunks_txt)
    
    # Complete exercises
    exercises(chunks_txt)
    
    print("=== Demo Complete ===")
    print("You have successfully explored various LangChain retrievers!")
    print("\nKey takeaways:")
    print("- Vector Store-Backed Retriever: Basic similarity search with various search types")
    print("- Multi-Query Retriever: Generates multiple query variations for comprehensive results")
    print("- Self-Querying Retriever: Automatically constructs structured queries from natural language")
    print("- Parent Document Retriever: Balances small chunks for accuracy with large chunks for context")

if __name__ == "__main__":
    main()
