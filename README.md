# Shakespeare RAG System 📚

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions about Shakespeare's works. It combines a retrieval system with a language model to provide accurate and context-aware answers.

> **Note**: This documentation is a work in progress. A more detailed setup guide will be published once the system is finalized.

## Features
- **Passage Retrieval**: Retrieves relevant passages from Shakespeare's works using FAISS and Sentence Transformers.
- **Answer Generation**: Generates answers using OpenAI's GPT or a custom LLaMA model.
- **Streamlit Interface**: Provides an interactive web interface for querying the system. (Currently for OpenAI API use only, please add your API to utils/openai.py to use the interface)
- **Evaluation**: Includes tools for evaluating the retrieval and generation components.



## Installation

### Prerequisites
- Python 3.10 or higher
- `pip` package manager
- OpenAI API key (if using GPT)

1. Clone the repository:
   ```bash
   git clone https://github.com/yunh-h/shakespeare_rag.git
   cd shakespeare_rag
   ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
Streamlit Interface:
```sh
streamlit run run.py
```
1. Open the Streamlit app in your browser (default: http://localhost:8502)

2. Enter a query, such as:
    ```
    Which sonnet has the line 'Shall I compare thee to a summer's day'?
    ```

## Key Components
1. Passage Retrieval    
    - Uses FAISS for efficient similarity search.
    - Sentence Transformers (all-MiniLM-L6-v2) for embedding queries and passages.
2. Answer Generation
    - Supports OpenAI's GPT models via openai.ChatCompletion.
    - Custom LLaMA pipeline for local inference.
3. Evaluation
    - Computes metrics like Recall@k and MRR for retrieval performance.
    - Compares generated answers with ground truth.
4. Streamlit Interface (Currently for OpenAI API use only)
    - Interactive web app for querying the system.
    - Displays retrieved passages and generated answers.

## Project Structure
```
shakespeare_rag/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── run.py                      # Streamlit app entry point
├── main.py                     # Core logic for loading resources and processing queries
├── eval.py                     # Evaluation script
├── data/                       # Data directory
│   ├── pg100.txt               # Original Shakespeare text
│   ├── all_chunks_400w40o.json # Chunked text data
│   └── all_chunks_400w40o_with_metadata.json # Chunked data with metadata
├── utils/                      # Utility modules
│   ├── retrieve.py             # Passage retrieval logic
│   ├── openai.py               # OpenAI GPT integration
│   ├── llama.py                # LLaMA model integration
│   ├── chunking.py             # Text chunking logic
│   └── data_processing.py      # Data preprocessing script
```

## Acknowledgments
- Project Gutenberg for Shakespeare's works.
