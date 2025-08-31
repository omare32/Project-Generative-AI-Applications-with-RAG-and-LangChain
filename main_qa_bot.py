#!/usr/bin/env python3
"""
Main QA Bot - Generative AI Applications with RAG and LangChain
A comprehensive chatbot that can load any PDF and answer questions using local models.

This script integrates all modules (m01, m02, m03) into a single, powerful QA system.
"""

import os
import sys
import torch
import gradio as gr
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add module paths to system path
sys.path.append(str(Path(__file__).parent / "m01"))
sys.path.append(str(Path(__file__).parent / "m02"))
sys.path.append(str(Path(__file__).parent / "m03"))

def check_gpu():
    """Check GPU availability and PyTorch configuration."""
    print("🔍 Checking GPU configuration...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("⚠️  No GPU detected. Using CPU (slower performance)")
        return False

def get_llm():
    """Get language model - tries Ollama first, falls back to HuggingFace."""
    try:
        # Try Ollama first (if installed)
        from langchain_community.llms import Ollama
        
        print("🔄 Attempting to load Ollama model...")
        llm = Ollama(model="llama2", temperature=0.1)
        
        # Test the model
        test_response = llm("Hello")
        if test_response:
            print("✅ Ollama model loaded successfully")
            return llm
            
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
    
    try:
        # Fallback to HuggingFace
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("🔄 Loading HuggingFace model as fallback...")
        model_name = "microsoft/DialoGPT-medium"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("✅ HuggingFace model loaded on GPU")
        else:
            print("✅ HuggingFace model loaded on CPU")
        
        # Create a simple wrapper
        class HuggingFaceLLM:
            def __call__(self, prompt):
                inputs = tokenizer.encode(prompt, return_tensors='pt')
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs, 
                        max_length=150, 
                        num_return_sequences=1,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response.replace(prompt, "").strip()
        
        return HuggingFaceLLM()
        
    except Exception as e:
        print(f"❌ Failed to load any language model: {e}")
        return None

def document_loader(file_path: str) -> Optional[List]:
    """Load PDF document using PyPDFLoader."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        
        print(f"📄 Loading PDF: {os.path.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        print(f"✅ Loaded {len(documents)} pages from PDF")
        return documents
        
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return None

def text_splitter(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> Optional[List]:
    """Split documents into manageable chunks using RecursiveCharacterTextSplitter."""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        print("✂️  Splitting documents into chunks...")
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        return None

def local_embedding():
    """Generate embeddings using local HuggingFace models."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        print("🧠 Loading embedding model...")
        
        # Use a local embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Check if GPU is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(f"✅ Loaded embedding model: {model_name} on {device.upper()}")
        return embeddings
        
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return None

def vector_database(chunks: List, embeddings) -> Optional[Any]:
    """Create Chroma vector database to store embeddings."""
    try:
        from langchain_community.vectorstores import Chroma
        
        print("🗄️  Creating vector database...")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        print(f"✅ Created vector database with {len(chunks)} chunks")
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        return None

def retriever(file_path: str, query: str, k: int = 4) -> Optional[List]:
    """Load, split, embed, and retrieve documents using ChromaDB."""
    try:
        print(f"🔍 Processing query: '{query}'")
        
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
        
        print(f"✅ Retrieved {len(docs)} relevant documents")
        return docs
        
    except Exception as e:
        print(f"❌ Error in retriever: {e}")
        return None

def qa_bot(file_path: str, query: str) -> str:
    """Main QA function that orchestrates the complete RAG pipeline."""
    try:
        print(f"\n🚀 Starting QA process for: {os.path.basename(file_path)}")
        print(f"❓ Query: {query}")
        
        # Get relevant documents
        docs = retriever(file_path, query)
        if not docs:
            return "❌ Error: Could not retrieve relevant documents. Please check if the PDF was loaded correctly."
        
        # Get LLM
        llm = get_llm()
        if not llm:
            return "❌ Error: Could not load language model. Please check your model installation."
        
        # Create QA chain
        from langchain.chains import RetrievalQA
        
        print("🔗 Creating QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever(file_path, query),
            return_source_documents=True
        )
        
        # Get answer
        print("💭 Generating answer...")
        result = qa_chain({"query": query})
        answer = result["result"]
        
        print("✅ Answer generated successfully")
        return answer
        
    except Exception as e:
        error_msg = f"❌ Error during QA process: {str(e)}"
        print(error_msg)
        return error_msg

def create_gradio_interface():
    """Create a beautiful Gradio interface for the QA Bot."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .upload-area {
        border: 2px dashed #667eea !important;
        border-radius: 10px !important;
    }
    .query-box {
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
    }
    .answer-box {
        background: #f8f9fa !important;
        border: 2px solid #28a745 !important;
        border-radius: 10px !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="RAG QA Bot - Generative AI Applications") as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🤖 RAG QA Bot</h1>
            <h3>Generative AI Applications with RAG and LangChain</h3>
            <p>Upload any PDF document and ask questions. Powered by local AI models and GPU acceleration.</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # File upload
                file_input = gr.File(
                    label="📄 Upload PDF Document",
                    file_types=[".pdf"],
                    file_count="single",
                    type="filepath"
                )
                
                # Query input
                query_input = gr.Textbox(
                    label="❓ Ask a Question",
                    placeholder="What is this document about? What are the main topics discussed?",
                    lines=3,
                    max_lines=5
                )
                
                # Submit button
                submit_btn = gr.Button("🚀 Get Answer", variant="primary", size="lg")
                
                # Example queries
                gr.Examples(
                    examples=[
                        ["What is this document about?"],
                        ["What are the main topics discussed?"],
                        ["Can you summarize the key points?"],
                        ["What are the conclusions of this paper?"],
                        ["Explain the methodology used in this research"]
                    ],
                    inputs=query_input,
                    label="💡 Example Questions"
                )
                
                # Status display
                status_display = gr.Textbox(
                    label="📊 Status",
                    value="Ready to process PDF documents and answer questions!",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=1):
                # Answer output
                answer_output = gr.Textbox(
                    label="💡 Answer",
                    placeholder="Your answer will appear here...",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )
                
                # File info
                file_info = gr.Textbox(
                    label="📋 File Information",
                    placeholder="File details will appear here...",
                    lines=3,
                    interactive=False
                )
        
        def process_query(file_path, query):
            """Process the query and return the answer."""
            if not file_path:
                return "❌ Please upload a PDF file first.", "No file uploaded"
            
            if not query.strip():
                return "❌ Please enter a question.", f"File: {os.path.basename(file_path)}"
            
            # Update status
            status = f"Processing: {os.path.basename(file_path)} | Query: {query[:50]}..."
            
            # Get answer
            answer = qa_bot(file_path, query)
            
            # Update file info
            file_info = f"File: {os.path.basename(file_path)}\nSize: {os.path.getsize(file_path) / 1024:.1f} KB\nStatus: Processed"
            
            return answer, file_info, "✅ Processing complete!"
        
        # Connect components
        submit_btn.click(
            fn=process_query,
            inputs=[file_input, query_input],
            outputs=[answer_output, file_info, status_display]
        )
        
        # Auto-process when file is uploaded
        file_input.change(
            fn=lambda x: f"✅ File uploaded: {os.path.basename(x) if x else 'No file'}" if x else "No file uploaded",
            inputs=file_input,
            outputs=status_display
        )
    
    return interface

def main():
    """Main function to run the QA Bot."""
    print("🚀 Starting RAG QA Bot - Generative AI Applications")
    print("=" * 60)
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Create and launch interface
    print("\n🎨 Creating Gradio interface...")
    interface = create_gradio_interface()
    
    print("\n🌐 Launching web interface...")
    print("📱 Open your browser and navigate to the local URL")
    print("📄 Upload any PDF document and start asking questions!")
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
