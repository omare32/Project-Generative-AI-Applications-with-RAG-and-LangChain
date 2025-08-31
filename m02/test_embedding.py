#!/usr/bin/env python3
"""
Simple test script for local embedding models
"""

import warnings
warnings.filterwarnings('ignore')

import os
import torch

def check_gpu():
    """Check GPU availability"""
    print("=== GPU Information ===\n")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Count: {gpu_count}")
        print(f"✓ Current Device: {current_device}")
        print(f"✓ CUDA Version: {torch.version.cuda}\n")
        return True
    else:
        print("⚠ No GPU available. Using CPU instead.\n")
        return False

def test_basic_embedding():
    """Test basic embedding functionality"""
    print("=== Testing Basic Embedding ===\n")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Use a smaller model for testing
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading model: {model_name}")
        
        model = SentenceTransformer(model_name, device=device)
        print("✓ Model loaded successfully")
        
        # Test with simple text
        test_text = "Hello, this is a test sentence."
        print(f"Testing with: '{test_text}'")
        
        embedding = model.encode(test_text)
        print(f"✓ Embedding generated successfully")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        return True
        
    except ImportError as e:
        print(f"⚠ Error importing SentenceTransformers: {e}")
        return False
    except Exception as e:
        print(f"⚠ Error with embedding: {e}")
        return False

def test_file_loading():
    """Test file loading functionality"""
    print("\n=== Testing File Loading ===\n")
    
    try:
        # Check if file exists
        filename = "state-of-the-union.txt"
        if os.path.exists(filename):
            print(f"✓ File exists: {filename}")
            
            # Try to read the file
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"✓ File read successfully")
                print(f"File size: {len(content)} characters")
                print(f"First 100 characters: {content[:100]}...")
                
            return True
        else:
            print(f"⚠ File not found: {filename}")
            return False
            
    except Exception as e:
        print(f"⚠ Error reading file: {e}")
        return False

def main():
    """Main test function"""
    print("=== Local Embedding Test ===\n")
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Test file loading
    file_ok = test_file_loading()
    
    # Test basic embedding
    embedding_ok = test_basic_embedding()
    
    print("\n=== Test Results ===")
    print(f"GPU Available: {'Yes' if gpu_available else 'No'}")
    print(f"File Loading: {'OK' if file_ok else 'Failed'}")
    print(f"Embedding: {'OK' if embedding_ok else 'Failed'}")
    
    if file_ok and embedding_ok:
        print("\n✓ All tests passed! You're ready to proceed.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
