#!/usr/bin/env python3
"""
Load Documents Using LangChain for Different Sources
==================================================

This script demonstrates how to use various loaders provided by LangChain to load and 
process data from different file formats. These loaders simplify the task of reading 
and converting files into a document format that can be processed downstream.

Objectives:
- Understand how to use TextLoader to load text files
- Learn how to load PDFs using PyPDFLoader and PyMuPDFLoader
- Use UnstructuredMarkdownLoader to load Markdown files
- Load JSON files with JSONLoader using jq schemas
- Process CSV files with CSVLoader and UnstructuredCSVLoader
- Load Webpage content using WebBaseLoader
- Load Word documents using Docx2txtLoader
- Utilize UnstructuredFileLoader for various file types
"""

import warnings
warnings.filterwarnings('ignore')

from pprint import pprint
import json
from pathlib import Path
import nltk
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredFileLoader
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

def load_from_txt_files():
    """Demonstrate loading from TXT files"""
    print("\n=== Loading from TXT files ===")
    
    # Download the text file
    txt_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Ec5f3KYU1CpbKRp1whFLZw/new-Policies.txt"
    download_file(txt_url, "new-Policies.txt")
    
    # Use TextLoader to load the file
    loader = TextLoader("new-Policies.txt")
    print(f"✓ TextLoader created: {loader}")
    
    # Load the data as documents
    data = loader.load()
    print(f"✓ Data loaded successfully. Number of documents: {len(data)}")
    
    # Display the document structure
    print("\nDocument structure:")
    print(f"Type: {type(data[0])}")
    print(f"Page content length: {len(data[0].page_content)} characters")
    print(f"Metadata: {data[0].metadata}")
    
    # Show first 1000 characters of content
    print("\nFirst 1000 characters of content:")
    pprint(data[0].page_content[:1000])
    
    return data

def load_from_pdf_files():
    """Demonstrate loading from PDF files"""
    print("\n=== Loading from PDF files ===")
    
    pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Q81D33CdRLK6LswuQrANQQ/instructlab.pdf"
    
    # PyPDFLoader
    print("\n--- PyPDFLoader ---")
    try:
        loader = PyPDFLoader(pdf_url)
        pages = loader.load_and_split()
        print(f"✓ PyPDFLoader loaded {len(pages)} pages")
        
        # Display first page
        print("\nFirst page:")
        print(pages[0])
        
        # Display first three pages
        print("\nFirst three pages:")
        for p, page in enumerate(pages[0:3]):
            print(f"Page number {p+1}")
            print(page)
            print("-" * 50)
    except Exception as e:
        print(f"⚠ Error with PyPDFLoader: {e}")
    
    # PyMuPDFLoader
    print("\n--- PyMuPDFLoader ---")
    try:
        loader = PyMuPDFLoader(pdf_url)
        print(f"✓ PyMuPDFLoader created: {loader}")
        
        data = loader.load()
        print(f"✓ PyMuPDFLoader loaded {len(data)} pages")
        
        # Display first page
        print("\nFirst page:")
        print(data[0])
        
        # Show metadata comparison
        print("\nPyMuPDFLoader provides more detailed metadata than PyPDFLoader")
    except Exception as e:
        print(f"⚠ Error with PyMuPDFLoader: {e}")

def load_from_markdown_files():
    """Demonstrate loading from Markdown files"""
    print("\n=== Loading from Markdown files ===")
    
    # Download markdown file
    md_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/eMSP5vJjj9yOfAacLZRWsg/markdown-sample.md'
    download_file(md_url, 'markdown-sample.md')
    
    # Use UnstructuredMarkdownLoader
    markdown_path = "markdown-sample.md"
    loader = UnstructuredMarkdownLoader(markdown_path)
    print(f"✓ UnstructuredMarkdownLoader created: {loader}")
    
    data = loader.load()
    print(f"✓ Markdown loaded successfully. Number of documents: {len(data)}")
    
    # Display the data
    print("\nMarkdown content:")
    print(data)

def load_from_json_files():
    """Demonstrate loading from JSON files"""
    print("\n=== Loading from JSON files ===")
    
    # Download JSON file
    json_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hAmzVJeOUAMHzmhUHNdAUg/facebook-chat.json'
    download_file(json_url, 'facebook-chat.json')
    
    # First, examine the JSON structure
    file_path = 'facebook-chat.json'
    data = json.loads(Path(file_path).read_text())
    
    print("JSON structure:")
    pprint(data)
    
    # Use JSONLoader with jq schema
    print("\n--- Using JSONLoader with jq schema ---")
    try:
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.messages[].content',
            text_content=False
        )
        
        data = loader.load()
        print(f"✓ JSON loaded successfully. Number of documents: {len(data)}")
        
        print("\nExtracted content:")
        pprint(data)
    except Exception as e:
        print(f"⚠ Error with JSONLoader: {e}")

def load_from_csv_files():
    """Demonstrate loading from CSV files"""
    print("\n=== Loading from CSV files ===")
    
    # Download CSV file
    csv_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IygVG_j0M87BM4Z0zFsBMA/mlb-teams-2012.csv'
    download_file(csv_url, 'mlb-teams-2012.csv')
    
    # CSVLoader
    print("\n--- CSVLoader ---")
    try:
        loader = CSVLoader(file_path='mlb-teams-2012.csv')
        data = loader.load()
        print(f"✓ CSV loaded successfully. Number of documents: {len(data)}")
        
        print("\nFirst few documents:")
        for i, doc in enumerate(data[:3]):
            print(f"Document {i+1}:")
            print(doc)
            print("-" * 30)
    except Exception as e:
        print(f"⚠ Error with CSVLoader: {e}")
    
    # UnstructuredCSVLoader
    print("\n--- UnstructuredCSVLoader ---")
    try:
        loader = UnstructuredCSVLoader(
            file_path="mlb-teams-2012.csv", 
            mode="elements"
        )
        data = loader.load()
        print(f"✓ UnstructuredCSV loaded successfully. Number of documents: {len(data)}")
        
        print("\nFirst document content:")
        print(data[0].page_content)
        
        print("\nHTML representation:")
        print(data[0].metadata["text_as_html"])
    except Exception as e:
        print(f"⚠ Error with UnstructuredCSVLoader: {e}")

def load_from_web_files():
    """Demonstrate loading from web pages"""
    print("\n=== Loading from URL/Website files ===")
    
    # BeautifulSoup limitation demonstration
    print("\n--- BeautifulSoup Limitation ---")
    try:
        import requests
        from bs4 import BeautifulSoup
        
        url = 'https://www.ibm.com/topics/langchain'
        response = requests.get(url)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        print("BeautifulSoup output (showing HTML tags and external links):")
        print(soup.prettify()[:500] + "...")
        print("\nBeautifulSoup loads HTML tags and external links, which may not be desired.")
    except Exception as e:
        print(f"⚠ Error with BeautifulSoup: {e}")
    
    # WebBaseLoader - single page
    print("\n--- WebBaseLoader (Single Page) ---")
    try:
        loader = WebBaseLoader("https://www.ibm.com/topics/langchain")
        data = loader.load()
        print(f"✓ Web page loaded successfully. Number of documents: {len(data)}")
        
        print("\nWeb content:")
        print(data)
    except Exception as e:
        print(f"⚠ Error with WebBaseLoader (single): {e}")
    
    # WebBaseLoader - multiple pages
    print("\n--- WebBaseLoader (Multiple Pages) ---")
    try:
        urls = [
            "https://www.ibm.com/topics/langchain", 
            "https://www.redhat.com/en/topics/ai/what-is-instructlab"
        ]
        loader = WebBaseLoader(urls)
        data = loader.load()
        print(f"✓ Multiple web pages loaded successfully. Number of documents: {len(data)}")
        
        print("\nMultiple web content:")
        print(data)
    except Exception as e:
        print(f"⚠ Error with WebBaseLoader (multiple): {e}")

def load_from_word_files():
    """Demonstrate loading from Word documents"""
    print("\n=== Loading from WORD files ===")
    
    # Download Word document
    docx_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/94hiHUNLZdb0bLMkrCh79g/file-sample.docx"
    download_file(docx_url, "file-sample.docx")
    
    # Use Docx2txtLoader
    try:
        loader = Docx2txtLoader("file-sample.docx")
        print(f"✓ Docx2txtLoader created: {loader}")
        
        data = loader.load()
        print(f"✓ Word document loaded successfully. Number of documents: {len(data)}")
        
        print("\nWord document content:")
        print(data)
    except Exception as e:
        print(f"⚠ Error with Docx2txtLoader: {e}")

def load_from_unstructured_files():
    """Demonstrate loading from unstructured files"""
    print("\n=== Loading from Unstructured Files ===")
    
    # Load .txt file
    print("\n--- Loading .txt file ---")
    try:
        loader = UnstructuredFileLoader("new-Policies.txt")
        data = loader.load()
        print(f"✓ UnstructuredFileLoader loaded .txt successfully. Number of documents: {len(data)}")
        print(data)
    except Exception as e:
        print(f"⚠ Error loading .txt with UnstructuredFileLoader: {e}")
    
    # Load .md file
    print("\n--- Loading .md file ---")
    try:
        loader = UnstructuredFileLoader("markdown-sample.md")
        data = loader.load()
        print(f"✓ UnstructuredFileLoader loaded .md successfully. Number of documents: {len(data)}")
        print(data)
    except Exception as e:
        print(f"⚠ Error loading .md with UnstructuredFileLoader: {e}")
    
    # Load multiple files with different formats
    print("\n--- Loading multiple files with different formats ---")
    try:
        files = ["markdown-sample.md", "new-Policies.txt"]
        loader = UnstructuredFileLoader(files)
        data = loader.load()
        print(f"✓ UnstructuredFileLoader loaded multiple files successfully. Number of documents: {len(data)}")
        print(data)
    except Exception as e:
        print(f"⚠ Error loading multiple files with UnstructuredFileLoader: {e}")

def exercises():
    """Complete the exercises from the notebook"""
    print("\n=== Exercises ===")
    
    # Exercise 1 - Try other PDF loaders
    print("\n--- Exercise 1: Try other PDF loaders ---")
    print("Testing PyPDFium2Loader...")
    try:
        # Install pypdfium2 if not available
        try:
            from langchain_community.document_loaders import PyPDFium2Loader
        except ImportError:
            print("Installing pypdfium2...")
            os.system("pip install pypdfium2")
            from langchain_community.document_loaders import PyPDFium2Loader
        
        pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Q81D33CdRLK6LswuQrANQQ/instructlab.pdf"
        loader = PyPDFium2Loader(pdf_url)
        data = loader.load()
        print(f"✓ PyPDFium2Loader loaded successfully. Number of documents: {len(data)}")
        print("PyPDFium2Loader provides another option for PDF processing")
    except Exception as e:
        print(f"⚠ Error with PyPDFium2Loader: {e}")
    
    # Exercise 2 - Load from Arxiv
    print("\n--- Exercise 2: Load from Arxiv ---")
    print("Testing ArxivLoader...")
    try:
        # Install arxiv if not available
        try:
            from langchain_community.document_loaders import ArxivLoader
        except ImportError:
            print("Installing arxiv...")
            os.system("pip install arxiv")
            from langchain_community.document_loaders import ArxivLoader
        
        docs = ArxivLoader(query="1605.08386", load_max_docs=2).load()
        print(f"✓ ArxivLoader loaded successfully. Number of documents: {len(docs)}")
        print("\nFirst document content (first 1000 characters):")
        print(docs[0].page_content[:1000])
    except Exception as e:
        print(f"⚠ Error with ArxivLoader: {e}")

def main():
    """Main function to demonstrate all document loaders"""
    print("=== LangChain Document Loader Demo ===\n")
    
    # Download NLTK data
    try:
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        print("✓ NLTK data downloaded successfully\n")
    except Exception as e:
        print(f"⚠ Error downloading NLTK data: {e}\n")
    
    # Demonstrate all loaders
    load_from_txt_files()
    load_from_pdf_files()
    load_from_markdown_files()
    load_from_json_files()
    load_from_csv_files()
    load_from_web_files()
    load_from_word_files()
    load_from_unstructured_files()
    
    # Complete exercises
    exercises()
    
    print("\n=== Demo Complete ===")
    print("You have successfully explored various LangChain document loaders!")
    print("These loaders provide a unified way to handle different file formats")
    print("for downstream processing in LLM applications.")

if __name__ == "__main__":
    main()
