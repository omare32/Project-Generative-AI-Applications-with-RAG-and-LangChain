#!/usr/bin/env python3
"""
Simple test script for the main QA bot
Tests basic functionality without requiring full dependencies.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if main modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import torch
        print("✅ PyTorch imported successfully")
        
        import gradio as gr
        print("✅ Gradio imported successfully")
        
        # Test LangChain imports
        from langchain_community.document_loaders import PyPDFLoader
        print("✅ PyPDFLoader imported successfully")
        
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ RecursiveCharacterTextSplitter imported successfully")
        
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✅ HuggingFaceEmbeddings imported successfully")
        
        from langchain_community.vectorstores import Chroma
        print("✅ Chroma imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_file_structure():
    """Test if required files and directories exist."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "main_qa_bot.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        ".gitignore"
    ]
    
    required_dirs = [
        "m01",
        "m02", 
        "m03"
    ]
    
    all_good = True
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_good = False
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ (missing)")
            all_good = False
    
    return all_good

def test_pdf_sample():
    """Test if sample PDF exists."""
    print("\n📄 Testing sample PDF...")
    
    pdf_path = Path("m03/document.to.load.pdf")
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"✅ Sample PDF found: {pdf_path}")
        print(f"   Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ Sample PDF not found: {pdf_path}")
        return False

def test_png_files():
    """Test if PNG submission files exist."""
    print("\n🖼️  Testing PNG submission files...")
    
    required_pngs = [
        "m03/pdf_loader.png",
        "m03/code_splitter.png",
        "m03/embedding.png",
        "m03/vectordb.png",
        "m03/retriever.png",
        "m03/QA_bot.png"
    ]
    
    all_good = True
    for png_path in required_pngs:
        if Path(png_path).exists():
            size_kb = Path(png_path).stat().st_size / 1024
            print(f"✅ {png_path} ({size_kb:.1f} KB)")
        else:
            print(f"❌ {png_path} (missing)")
            all_good = False
    
    return all_good

def test_requirements():
    """Test if requirements.txt is valid."""
    print("\n📋 Testing requirements.txt...")
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
            
        if "langchain" in content and "torch" in content:
            print("✅ requirements.txt contains required packages")
            return True
        else:
            print("❌ requirements.txt missing required packages")
            return False
            
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Generative AI Applications with RAG and LangChain")
    print("=" * 70)
    
    tests = [
        ("Import Tests", test_imports),
        ("File Structure", test_file_structure),
        ("Sample PDF", test_pdf_sample),
        ("PNG Files", test_png_files),
        ("Requirements", test_requirements)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Run: python main_qa_bot.py")
        print("   2. Open browser to http://localhost:7860")
        print("   3. Upload a PDF and start asking questions!")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\n💡 To fix issues:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Check file paths and permissions")
        print("   3. Ensure all required files are present")

if __name__ == "__main__":
    main()
