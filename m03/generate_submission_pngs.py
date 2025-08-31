#!/usr/bin/env python3
"""
Generate submission PNG files for the QA Bot project.
This script creates PNG files showing code snippets for each required task.
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import ImageFormatter
    from pygments.styles import get_style_by_name
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Installing required libraries...")
    os.system("pip install Pillow pygments")
    from PIL import Image, ImageDraw, ImageFont
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import ImageFormatter
    from pygments.styles import get_style_by_name

def create_code_snippet_image(code, filename, title="Code Snippet", width=800, height=600):
    """Create a PNG image showing code with syntax highlighting."""
    
    # Create a style for the code
    style = get_style_by_name('monokai')
    
    # Configure the image formatter
    formatter = ImageFormatter(
        style=style,
        image_format='PNG',
        image_pad=20,
        line_numbers=True,
        font_size=14,
        line_number_chars=3,
        line_number_bg=style.background_color,
        line_number_fg='#888888'
    )
    
    # Highlight the code
    highlighted_code = highlight(code, PythonLexer(), formatter)
    
    # Save the image
    with open(filename, 'wb') as f:
        f.write(highlighted_code)
    
    print(f"Created {filename}")

def create_simple_code_image(code, filename, title="Code Snippet"):
    """Create a simple PNG image with code (fallback if pygments fails)."""
    
    # Create a new image with white background
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a system font, fallback to default
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Draw title
    draw.text((20, 20), title, fill='black', font=font)
    
    # Draw code with basic formatting
    lines = code.split('\n')
    y_position = 60
    
    for i, line in enumerate(lines):
        if y_position > 550:  # Don't go off the image
            break
        
        # Add line numbers
        line_num = f"{i+1:2d}: "
        draw.text((20, y_position), line_num, fill='blue', font=font)
        
        # Add code
        draw.text((80, y_position), line, fill='black', font=font)
        y_position += 20
    
    # Save the image
    img.save(filename)
    print(f"Created {filename}")

def generate_all_pngs():
    """Generate all required PNG files for submission."""
    
    # Task 1: PDF Loader
    pdf_loader_code = '''def document_loader(file_path):
    """Load PDF document using PyPDFLoader."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        
        # Load the PDF file
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages from PDF")
        return documents
        
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None'''
    
    create_code_snippet_image(pdf_loader_code, "pdf_loader.png", "PDF Loader Function")
    
    # Task 2: Text Splitter
    text_splitter_code = '''def text_splitter(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into manageable chunks using RecursiveCharacterTextSplitter."""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\\n\\n", "\\n", " ", ""]
        )
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return None'''
    
    create_code_snippet_image(text_splitter_code, "code_splitter.png", "Text Splitter Function")
    
    # Task 3: Embedding Model
    embedding_code = '''def local_embedding():
    """Generate embeddings using local HuggingFace models."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Use a local embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(f"Loaded embedding model: {model_name}")
        return embeddings
        
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None'''
    
    create_code_snippet_image(embedding_code, "embedding.png", "Embedding Model Function")
    
    # Task 4: Vector Database
    vectordb_code = '''def vector_database(chunks, embeddings):
    """Create Chroma vector database to store embeddings."""
    try:
        from langchain_community.vectorstores import Chroma
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        print(f"Created vector database with {len(chunks)} chunks")
        return vectorstore
        
    except Exception as e:
        print(f"Error creating vector database: {e}")
        return None'''
    
    create_code_snippet_image(vectordb_code, "vectordb.png", "Vector Database Function")
    
    # Task 5: Retriever
    retriever_code = '''def retriever(file_path, query, k=4):
    """Load, split, embed, and retrieve documents using ChromaDB."""
    try:
        # Load document
        documents = document_loader(file_path)
        if not documents:
            return None
        
        # Split into chunks
        chunks = text_splitter(documents)
        if not chunks:
            return None
        
        # Get embeddings
        embeddings = local_embedding()
        if not embeddings:
            return None
        
        # Create vector database
        vectorstore = vector_database(chunks, embeddings)
        if not vectorstore:
            return None
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query)
        
        print(f"Retrieved {len(docs)} relevant documents")
        return docs
        
    except Exception as e:
        print(f"Error in retriever: {e}")
        return None'''
    
    create_code_snippet_image(retriever_code, "retriever.png", "Retriever Function")
    
    # Task 6: QA Bot Interface
    qa_bot_code = '''def create_gradio_interface():
    """Create Gradio interface for the QA Bot."""
    import gradio as gr
    
    def qa_bot(file_path, query):
        """Main QA function."""
        try:
            # Get relevant documents
            docs = retriever(file_path, query)
            if not docs:
                return "Error: Could not retrieve relevant documents."
            
            # Get LLM
            llm = get_llm()
            if not llm:
                return "Error: Could not load language model."
            
            # Create QA chain
            from langchain.chains import RetrievalQA
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever(file_path, query),
                return_source_documents=True
            )
            
            # Get answer
            result = qa_chain({"query": query})
            answer = result["result"]
            
            return answer
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=qa_bot,
        inputs=[
            gr.File(label="Upload PDF File", file_types=[".pdf"]),
            gr.Textbox(label="Query", placeholder="What is this paper about?")
        ],
        outputs=gr.Textbox(label="Answer", lines=10),
        title="RAG QA Bot",
        description="Upload a PDF and ask questions about its content."
    )
    
    return interface'''
    
    create_code_snippet_image(qa_bot_code, "QA_bot.png", "QA Bot Interface")

def main():
    """Main function to generate all PNG files."""
    print("Generating submission PNG files...")
    
    try:
        generate_all_pngs()
        print("\\nAll PNG files generated successfully!")
        print("Files created:")
        print("- pdf_loader.png")
        print("- code_splitter.png") 
        print("- embedding.png")
        print("- vectordb.png")
        print("- retriever.png")
        print("- QA_bot.png")
        
    except Exception as e:
        print(f"Error generating PNG files: {e}")
        print("Trying fallback method...")
        
        # Fallback: create simple images
        try:
            generate_all_pngs_fallback()
            print("Fallback PNG files created successfully!")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

def generate_all_pngs_fallback():
    """Fallback method using simple image creation."""
    
    # Simple code snippets for fallback
    codes = {
        "pdf_loader.png": "def document_loader(file_path):\n    loader = PyPDFLoader(file_path)\n    return loader.load()",
        "code_splitter.png": "def text_splitter(docs):\n    splitter = RecursiveCharacterTextSplitter()\n    return splitter.split_documents(docs)",
        "embedding.png": "def local_embedding():\n    return HuggingFaceEmbeddings()",
        "vectordb.png": "def vector_database(chunks, embeddings):\n    return Chroma.from_documents(chunks, embeddings)",
        "retriever.png": "def retriever(file_path, query):\n    docs = load_and_split(file_path)\n    return vectorstore.as_retriever()",
        "QA_bot.png": "def create_gradio_interface():\n    return gr.Interface(fn=qa_bot, inputs=[...], outputs=...)"
    }
    
    for filename, code in codes.items():
        create_simple_code_image(code, filename, filename.replace('.png', '').replace('_', ' ').title())

if __name__ == "__main__":
    main()
