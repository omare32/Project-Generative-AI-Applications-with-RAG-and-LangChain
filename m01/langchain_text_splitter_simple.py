#!/usr/bin/env python3
"""
Apply Text Splitting Techniques to Enhance Model Responsiveness
=============================================================

This script demonstrates how to use commonly used text splitters from LangChain to 
split your source documents into chunks for downstream use in RAG (Retrieval-Augmented Generation).

Objectives:
- Use commonly used text splitters from LangChain
- Split source documents into chunks for downstream use in RAG
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

def demonstrate_document_object():
    """Demonstrate the Document object in LangChain"""
    print("\n=== Document Object in LangChain ===")
    
    try:
        from langchain_core.documents import Document
        
        # Create a document object example
        doc = Document(
            page_content="""Python is an interpreted high-level general-purpose programming language. 
                            Python's design philosophy emphasizes code readability with its notable use of significant indentation.""",
            metadata={
                'my_document_id': 234234,
                'my_document_source': "About Python",
                'my_document_create_time': 1680013019
            }
        )
        
        print("Document object created:")
        print(f"Page content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("✓ Document object demonstrates the structure used by LangChain")
        
    except ImportError as e:
        print(f"⚠ Error importing Document: {e}")
        print("Please install langchain-core: pip install langchain-core")

def split_by_character():
    """Demonstrate splitting by character"""
    print("\n=== Split by Character ===")
    
    try:
        from langchain.text_splitter import CharacterTextSplitter
        
        # Load the company policies document
        with open("companypolicies.txt") as f:
            companypolicies = f.read()
        
        print(f"Document length: {len(companypolicies)} characters")
        
        # Create CharacterTextSplitter
        text_splitter = CharacterTextSplitter(
            separator="",
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )
        
        print("CharacterTextSplitter created with:")
        print("- Separator: '' (empty string)")
        print("- Chunk size: 200")
        print("- Chunk overlap: 20")
        print("- Length function: len")
        
        # Split the text
        texts = text_splitter.split_text(companypolicies)
        print(f"\n✓ Text split successfully into {len(texts)} chunks")
        
        # Show first few chunks
        print("\nFirst 3 chunks:")
        for i, chunk in enumerate(texts[:3]):
            print(f"Chunk {i+1}: {chunk[:100]}...")
        
        # Create documents with metadata
        print("\n--- Creating documents with metadata ---")
        texts_with_metadata = text_splitter.create_documents(
            [companypolicies], 
            metadatas=[{"document": "Company Policies"}]
        )
        
        print(f"✓ Created {len(texts_with_metadata)} document objects")
        print(f"First document: {texts_with_metadata[0]}")
        
        return texts
        
    except ImportError as e:
        print(f"⚠ Error importing CharacterTextSplitter: {e}")
        print("Please install langchain: pip install langchain")
    except Exception as e:
        print(f"⚠ Error with character splitting: {e}")

def split_recursively_by_character():
    """Demonstrate recursive character splitting"""
    print("\n=== Recursively Split by Character ===")
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Load the company policies document
        with open("companypolicies.txt") as f:
            companypolicies = f.read()
        
        # Create RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )
        
        print("RecursiveCharacterTextSplitter created with:")
        print("- Default separators: ['\\n\\n', '\\n', ' ', '']")
        print("- Chunk size: 100")
        print("- Chunk overlap: 20")
        print("- Length function: len")
        
        # Split the text
        texts = text_splitter.create_documents([companypolicies])
        print(f"\n✓ Text split recursively into {len(texts)} chunks")
        
        # Show first few chunks
        print("\nFirst 3 chunks:")
        for i, chunk in enumerate(texts[:3]):
            print(f"Chunk {i+1}: {chunk.page_content[:100]}...")
        
        return texts
        
    except ImportError as e:
        print(f"⚠ Error importing RecursiveCharacterTextSplitter: {e}")
        print("Please install langchain: pip install langchain")
    except Exception as e:
        print(f"⚠ Error with recursive splitting: {e}")

def split_code():
    """Demonstrate splitting code by language"""
    print("\n=== Split Code ===")
    
    try:
        from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
        
        # Show supported languages
        print("Supported programming languages:")
        supported_languages = [e.value for e in Language]
        print(supported_languages)
        
        # Python code splitting
        print("\n--- Python Code Splitting ---")
        PYTHON_CODE = """
        def hello_world():
            print("Hello, World!")
        
        # Call the function
        hello_world()
        """
        
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=50, 
            chunk_overlap=0
        )
        
        python_docs = python_splitter.create_documents([PYTHON_CODE])
        print(f"✓ Python code split into {len(python_docs)} chunks")
        
        for i, doc in enumerate(python_docs):
            print(f"Chunk {i+1}: {doc.page_content}")
        
        # JavaScript code splitting
        print("\n--- JavaScript Code Splitting ---")
        JS_CODE = """
        function helloWorld() {
          console.log("Hello, World!");
        }
        
        // Call the function
        helloWorld();
        """
        
        js_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS, 
            chunk_size=60, 
            chunk_overlap=0
        )
        
        js_docs = js_splitter.create_documents([JS_CODE])
        print(f"✓ JavaScript code split into {len(js_docs)} chunks")
        
        for i, doc in enumerate(js_docs):
            print(f"Chunk {i+1}: {doc.page_content}")
        
        return python_docs, js_docs
        
    except ImportError as e:
        print(f"⚠ Error importing code splitters: {e}")
        print("Please install langchain: pip install langchain")
    except Exception as e:
        print(f"⚠ Error with code splitting: {e}")

def split_markdown_by_headers():
    """Demonstrate splitting markdown by headers"""
    print("\n=== Markdown Header Text Splitter ===")
    
    try:
        from langchain.text_splitter import MarkdownHeaderTextSplitter
        
        # Sample markdown text
        md = """# Foo

## Bar

Hi this is Jim

Hi this is Joe

### Boo 

Hi this is Lance 

## Baz

Hi this is Molly"""
        
        print("Sample markdown:")
        print(md)
        
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        print(f"\nHeaders to split on: {headers_to_split_on}")
        
        # Split with headers stripped
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(md)
        
        print(f"\n✓ Markdown split into {len(md_header_splits)} chunks (headers stripped)")
        
        for i, split in enumerate(md_header_splits):
            print(f"Chunk {i+1}:")
            print(f"  Content: {split.page_content}")
            print(f"  Metadata: {split.metadata}")
            print()
        
        # Split with headers included
        print("--- Including headers in content ---")
        markdown_splitter_with_headers = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False
        )
        md_header_splits_with_headers = markdown_splitter_with_headers.split_text(md)
        
        print(f"✓ Markdown split into {len(md_header_splits_with_headers)} chunks (headers included)")
        
        for i, split in enumerate(md_header_splits_with_headers):
            print(f"Chunk {i+1}:")
            print(f"  Content: {split.page_content}")
            print(f"  Metadata: {split.metadata}")
            print()
        
        return md_header_splits
        
    except ImportError as e:
        print(f"⚠ Error importing MarkdownHeaderTextSplitter: {e}")
        print("Please install langchain: pip install langchain")
    except Exception as e:
        print(f"⚠ Error with markdown splitting: {e}")

def exercises():
    """Complete the exercises from the notebook"""
    print("\n=== Exercises ===")
    
    # Exercise 1 - Changing separator for CharacterTextSplitter
    print("\n--- Exercise 1: Changing separator for CharacterTextSplitter ---")
    print("Testing with '\\n' separator...")
    
    try:
        from langchain.text_splitter import CharacterTextSplitter
        
        with open("companypolicies.txt") as f:
            companypolicies = f.read()
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )
        
        texts = text_splitter.split_text(companypolicies)
        print(f"✓ Text split with '\\n' separator into {len(texts)} chunks")
        
        print("\nFirst 3 chunks with '\\n' separator:")
        for i, chunk in enumerate(texts[:3]):
            print(f"Chunk {i+1}: {chunk[:100]}...")
            
    except Exception as e:
        print(f"⚠ Error with exercise 1: {e}")
    
    # Exercise 2 - Splitting Latex code
    print("\n--- Exercise 2: Splitting Latex code ---")
    
    try:
        from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
        
        latex_text = """
        \\documentclass{article}
        
        \\begin{document}
        
        \\maketitle
        
        \\section{Introduction}
        Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in a variety of natural language processing tasks, including language translation, text generation, and sentiment analysis.
        
        \\subsection{History of LLMs}
        The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.
        
        \\subsection{Applications of LLMs}
        LLMs have many applications in industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.
        
        \\end{document}
        """
        
        print("Sample LaTeX text:")
        print(latex_text)
        
        latex_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.LATEX, 
            chunk_size=60, 
            chunk_overlap=0
        )
        
        latex_docs = latex_splitter.create_documents([latex_text])
        print(f"\n✓ LaTeX code split into {len(latex_docs)} chunks")
        
        for i, doc in enumerate(latex_docs):
            print(f"Chunk {i+1}: {doc.page_content}")
        
    except Exception as e:
        print(f"⚠ Error with exercise 2: {e}")

def main():
    """Main function to demonstrate all text splitters"""
    print("=== LangChain Text Splitter Demo ===\n")
    
    # Download the company policies document
    if not download_file(
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YRYau14UJyh0DdiLDdzFcA/companypolicies.txt",
        "companypolicies.txt"
    ):
        print("⚠ Cannot proceed without the sample document. Please download manually or check your connection.")
        return
    
    # Load and display the document
    print("\n=== Document Overview ===")
    try:
        with open("companypolicies.txt") as f:
            companypolicies = f.read()
        
        print(f"Document loaded successfully. Length: {len(companypolicies)} characters")
        print("This is a long document about a company's policies that we'll use for splitting demonstrations.\n")
    except FileNotFoundError:
        print("⚠ Document file not found. Cannot proceed with demonstrations.")
        return
    
    # Demonstrate all text splitters
    demonstrate_document_object()
    split_by_character()
    split_recursively_by_character()
    split_code()
    split_markdown_by_headers()
    
    # Complete exercises
    exercises()
    
    print("\n=== Demo Complete ===")
    print("You have successfully explored various LangChain text splitters!")
    print("These splitters help break down large documents into manageable chunks")
    print("for efficient processing in RAG applications.")

if __name__ == "__main__":
    main()
