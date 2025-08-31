#!/usr/bin/env python3
"""
Demo script for the RAG QA Bot
This script demonstrates how to use the main QA bot programmatically.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def demo_qa_bot():
    """Demonstrate the QA bot functionality."""
    
    print("🚀 RAG QA Bot Demo")
    print("=" * 50)
    
    # Check if sample PDF exists
    sample_pdf = Path("m03/document.to.load.pdf")
    if not sample_pdf.exists():
        print("❌ Sample PDF not found. Please ensure m03/document.to.load.pdf exists.")
        return
    
    print(f"✅ Found sample PDF: {sample_pdf}")
    
    try:
        # Import the main QA bot functions
        from main_qa_bot import check_gpu, document_loader, text_splitter, local_embedding
        
        # Check GPU
        print("\n🔍 Checking GPU configuration...")
        gpu_available = check_gpu()
        
        # Load document
        print(f"\n📄 Loading sample PDF...")
        documents = document_loader(str(sample_pdf))
        if not documents:
            print("❌ Failed to load PDF")
            return
        
        print(f"✅ Loaded {len(documents)} pages")
        
        # Split text
        print(f"\n✂️  Splitting text into chunks...")
        chunks = text_splitter(documents, chunk_size=500, chunk_overlap=100)
        if not chunks:
            print("❌ Failed to split text")
            return
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Show sample chunks
        print(f"\n📝 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1}: {chunk.page_content[:100]}...")
        
        # Test embedding
        print(f"\n🧠 Testing embedding model...")
        embeddings = local_embedding()
        if not embeddings:
            print("❌ Failed to load embedding model")
            return
        
        print("✅ Embedding model loaded successfully")
        
        # Test with a sample query
        print(f"\n❓ Testing with sample query...")
        from main_qa_bot import qa_bot
        
        query = "What is this document about?"
        print(f"Query: {query}")
        
        answer = qa_bot(str(sample_pdf), query)
        print(f"Answer: {answer}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"💡 Try running 'python main_qa_bot.py' for the full web interface!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during demo: {e}")

def demo_modules():
    """Demonstrate individual modules."""
    
    print("\n🔧 Module Demonstrations")
    print("=" * 30)
    
    modules = [
        ("m01", "Document Processing"),
        ("m02", "Embeddings & Vector DB"),
        ("m03", "QA Bot Implementation")
    ]
    
    for module_path, description in modules:
        if Path(module_path).exists():
            print(f"✅ {module_path}/ - {description}")
            
            # List Python files
            py_files = list(Path(module_path).glob("*.py"))
            if py_files:
                print(f"   📄 Python files: {len(py_files)}")
                for py_file in py_files[:3]:  # Show first 3
                    print(f"      - {py_file.name}")
                if len(py_files) > 3:
                    print(f"      ... and {len(py_files) - 3} more")
        else:
            print(f"❌ {module_path}/ - {description} (not found)")

def main():
    """Main demo function."""
    
    print("🤖 Generative AI Applications with RAG and LangChain")
    print("=" * 60)
    
    # Show project structure
    print("\n📁 Project Structure:")
    print("├── m01/ - Document Processing")
    print("├── m02/ - Embeddings & Vector DB")
    print("├── m03/ - QA Bot Implementation")
    print("├── main_qa_bot.py - Main integrated bot")
    print("└── demo.py - This demo script")
    
    # Demo modules
    demo_modules()
    
    # Demo QA bot
    demo_qa_bot()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run main bot: python main_qa_bot.py")
    print("3. Upload PDFs and start asking questions!")
    print("4. Check out individual modules in m01/, m02/, m03/")

if __name__ == "__main__":
    main()
