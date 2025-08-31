#!/usr/bin/env python3
"""
Full Document Retrieve Limitation
================================

This script demonstrates the concept of context length for LLMs and the limitations 
of retrieving information when inputting the entire content of a document into a prompt.

Objectives:
- Explain the concept of context length for LLMs
- Recognize the limitations of retrieving information when inputting the entire content 
  of a document into a prompt
"""

import warnings
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_ibm import WatsonxLLM
import os

def llm_model(model_id):
    """
    Create a function that interacts with the watsonx.ai API, enabling you to utilize 
    various models available.
    
    Args:
        model_id (str): The model ID to use
        
    Returns:
        WatsonxLLM: LLM object that can be used to invoke queries
    """
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,  # controls the maximum number of tokens in the generated output
        GenParams.TEMPERATURE: 0.5,     # randomness or creativity of the model's responses
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
    
    llm = WatsonxLLM(watsonx_model=model)
    return llm

def main():
    """Main function to demonstrate document retrieval limitations"""
    
    print("=== Full Document Retrieve Limitation Demo ===\n")
    
    # Build LLM - try to invoke an example query
    print("1. Building LLM and testing with example query...")
    try:
        llama_llm = llm_model('meta-llama/llama-3-3-70b-instruct')
        response = llama_llm.invoke("How are you?")
        print(f"LLM Response: {response}")
        print("✓ LLM is working correctly\n")
    except Exception as e:
        print(f"⚠ Error with LLM: {e}")
        print("Note: You may need to set up proper credentials for watsonx.ai\n")
        return
    
    # Load source document
    print("2. Loading source document...")
    
    # Download the document if it doesn't exist
    if not os.path.exists("state-of-the-union.txt"):
        print("Downloading state-of-the-union.txt...")
        try:
            import urllib.request
            urllib.request.urlretrieve(
                "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/d_ahNwb1L2duIxBR6RD63Q/state-of-the-union.txt",
                "state-of-the-union.txt"
            )
            print("✓ state-of-the-union.txt downloaded successfully")
        except Exception as e:
            print(f"⚠ Error downloading document: {e}")
            print("Please download manually or check your internet connection")
            return
    
    # Use TextLoader to load the text
    try:
        loader = TextLoader("state-of-the-union.txt")
        data = loader.load()
        content = data[0].page_content
        print(f"✓ Document loaded successfully. Length: {len(content)} characters")
        print(f"  Document contains approximately 8,235 tokens\n")
    except Exception as e:
        print(f"⚠ Error loading document: {e}")
        return
    
    # Demonstrate limitations of retrieving directly from full document
    print("3. Demonstrating limitations of retrieving directly from full document...")
    
    # Context length explanation
    print("Context Length:")
    print("- LLMs have a fixed context length (GPT-3: 4096 tokens, GPT-4: 8192 tokens)")
    print("- Your document is 8,235 tokens, which exceeds most model limits")
    print("- This necessitates chunking strategies for effective retrieval\n")
    
    # LangChain prompt template
    print("LangChain Prompt Template:")
    template = """According to the document content here 
                {content},
                answer this question 
                {question}.
                Do not try to make up the answer.
                    
                YOUR RESPONSE:
    """
    
    prompt_template = PromptTemplate(template=template, input_variables=['content', 'question'])
    print("✓ Prompt template created successfully\n")
    
    # Use mixtral model (longer context window)
    print("4. Testing with Mixtral model (longer context window)...")
    try:
        mixtral_llm = llm_model('mistralai/mixtral-8x7b-instruct-v01')
        query_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template)
        
        query = "It is in which year of our nation?"
        response = query_chain.invoke(input={'content': content, 'question': query})
        print(f"Query: {query}")
        print(f"Response: {response['text']}")
        print("✓ Mixtral model successfully processed the full document\n")
    except Exception as e:
        print(f"⚠ Error with Mixtral model: {e}")
    
    # Use Llama 3 model
    print("5. Testing with Llama 3 model...")
    try:
        query_chain = LLMChain(llm=llama_llm, prompt=prompt_template)
        
        query = "It is in which year of our nation?"
        response = query_chain.invoke(input={'content': content, 'question': query})
        print(f"Query: {query}")
        print(f"Response: {response['text']}")
        print("✓ Llama 3 model also successfully processed the full document\n")
    except Exception as e:
        print(f"⚠ Error with Llama 3 model: {e}")
    
    # Key takeaway
    print("6. Key Takeaway:")
    print("If the document is much longer than the LLM's context length, it is important")
    print("and necessary to cut the document into chunks, index them, and then let the")
    print("LLM retrieve the relevant information accurately and efficiently.\n")
    
    # Exercise 1 - Try another LLM
    print("7. Exercise 1 - Testing with IBM Granite model...")
    try:
        granite_llm = llm_model('ibm/granite-3-8b-instruct')
        query_chain = LLMChain(llm=granite_llm, prompt=prompt_template)
        
        query = "It is in which year of our nation?"
        response = query_chain.invoke(input={'content': content, 'question': query})
        print(f"Query: {query}")
        print(f"Response: {response['text']}")
        print("✓ IBM Granite model test completed\n")
    except Exception as e:
        print(f"⚠ Error with IBM Granite model: {e}")
        print("This demonstrates that different models may have different capabilities\n")
    
    print("=== Demo Complete ===")
    print("In the next lesson, you will learn how to perform document chunking,")
    print("indexing, and retrieval using LangChain.")

if __name__ == "__main__":
    main()
