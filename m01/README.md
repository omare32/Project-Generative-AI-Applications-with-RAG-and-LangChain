# Module 01: Document Processing Fundamentals

This module covers the foundational concepts of document processing for RAG (Retrieval-Augmented Generation) applications using LangChain.

## Overview

Module 01 introduces three key concepts:
1. **Document Retrieval Limitations** - Understanding LLM context length constraints
2. **Document Loading** - Using LangChain loaders for various file formats
3. **Text Splitting** - Breaking documents into manageable chunks for processing

## Files

### 1. `full_document_retrieve_limitation.py`
**Purpose**: Demonstrates the limitations of retrieving information from full documents due to LLM context length constraints.

**Key Concepts**:
- Context length limitations in LLMs (GPT-3: 4096 tokens, GPT-4: 8192 tokens)
- Challenges of processing long documents (8,235+ tokens)
- Testing different LLM models (Mixtral, Llama 3, IBM Granite)
- Necessity of document chunking strategies

**Learning Objectives**:
- Explain the concept of context length for LLMs
- Recognize limitations of inputting entire documents into prompts

### 2. `langchain_document_loader.py`
**Purpose**: Demonstrates how to use various LangChain document loaders for different file formats.

**Supported Formats**:
- **Text files**: `TextLoader`
- **PDF files**: `PyPDFLoader`, `PyMuPDFLoader`
- **Markdown files**: `UnstructuredMarkdownLoader`
- **JSON files**: `JSONLoader` with jq schemas
- **CSV files**: `CSVLoader`, `UnstructuredCSVLoader`
- **Web pages**: `WebBaseLoader`
- **Word documents**: `Docx2txtLoader`
- **Unstructured files**: `UnstructuredFileLoader`

**Learning Objectives**:
- Understand how to use `TextLoader` for text files
- Learn PDF loading with different loaders
- Process various file formats into unified document structures
- Handle multiple file types efficiently

### 3. `langchain_text_splitter.py`
**Purpose**: Demonstrates how to use various text splitters from LangChain to split source documents into chunks for RAG applications.

**Text Splitters Covered**:
- **Character-based**: `CharacterTextSplitter` - simple character-based splitting
- **Recursive**: `RecursiveCharacterTextSplitter` - intelligent splitting preserving semantic units
- **Code-aware**: Language-specific splitters for Python, JavaScript, LaTeX, etc.
- **Structure-aware**: `MarkdownHeaderTextSplitter`, `HTMLHeaderTextSplitter`, `HTMLSectionSplitter`

**Key Parameters**:
- `chunk_size`: Maximum size of chunks
- `chunk_overlap`: Overlap between consecutive chunks
- `separator`: Characters used for splitting
- `length_function`: How chunk length is calculated

**Learning Objectives**:
- Use commonly used text splitters from LangChain
- Split source documents into chunks for downstream RAG use

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up IBM Watson AI credentials** (for LLM functionality):
   - You'll need access to IBM's watsonx.ai platform
   - Set up your project ID and credentials

## Usage

### Running Individual Scripts

```bash
# Run the document retrieval limitation demo
python full_document_retrieve_limitation.py

# Run the document loader demo
python langchain_document_loader.py

# Run the text splitter demo
python langchain_text_splitter.py
```

### Running All Scripts

```bash
# Run all demonstrations in sequence
python full_document_retrieve_limitation.py
python langchain_document_loader.py
python langchain_text_splitter.py
```

## Key Takeaways

1. **Context Length Awareness**: LLMs have fixed context windows that limit how much text they can process at once.

2. **Document Loading Strategy**: LangChain provides unified loaders for various file formats, making it easy to process different document types.

3. **Text Splitting Importance**: Breaking large documents into manageable chunks is essential for effective RAG applications.

4. **Metadata Preservation**: Document loaders and splitters maintain important metadata for downstream processing.

## Next Steps

After completing this module, you'll be ready to:
- Move to Module 02: Document embedding and vector storage
- Implement document chunking strategies in your RAG applications
- Handle various document formats in production environments

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed from `requirements.txt`
- **LLM Errors**: Check your IBM Watson AI credentials and project setup
- **File Download Issues**: Some scripts download sample files; ensure internet connectivity
- **Memory Issues**: Large documents may require significant memory; adjust chunk sizes accordingly

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [IBM Watson AI Documentation](https://ibm.github.io/watsonx-ai-python-sdk/)
- [Text Splitting Best Practices](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
