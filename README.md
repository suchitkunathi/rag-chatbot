# Rag-Chatbot
Multiple PDFs RAG chatbot

## How to Run Locally (Windows)

Similar steps for **MacOS** or **Linux** 

Step 1 :
[Install Poppler](https://github.com/oschwartz10612/poppler-windows/releases)

-    Add bin directory to PATH in System Variables

Step 2 :
[Install Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki)

-   Add folder "Tesseract-OCR" to PATH in System Variables

Step 3 : [Install Ollama](https://ollama.com/download)
```bash
ollama pull mistral:7b
```
- Depending on your GPU VRAM you can download your compatible LLM
```bash
ollama serve
```
- Start Ollama using the above command before running 


Step 3 : Clone Repository in your Machine
```bash
git clone https://github.com/suchitkunathi/rag-chatbot.git
```
```bash
cd rag-chatbot
```

Step 4 : Create Virtual Environment and Activate
```bash
conda create -p venv python==3.13 -y
```

```bash
conda activate venv/
```
Note : Not necessary to use Anaconda for virtual environment but it is recommended

Step 5 :
Install Requirements
```bash
pip install -r requirements.txt
```
    Requirements :
    - langchain
    - langchain-community
    - langchain-ollama
    - langchain-text-splitters
    - langchain-huggingface
    - streamlit
    - unstructured[pdf] 
    - sentence-transformers
    - faiss-cpu

Note : LangChain keeps deprecating old imports so incase of any import errors refer to [LangChain Documentation](https://docs.langchain.com/oss/python/langchain/overview)

Step 6 : Run Streamlit App

```bash
streamlit run main.py
```

## Important Notes

- Included Comments in Code for better Readability

Built for a Nvidia RTX 4060 8GB VRAM. (but anyone can run it depending on the llm used)

LLm used is Mistral 7B which is 14GB is quantized down to 4GB (4-bit) by Ollama (GGUF format) comfortably fitting the 8GB VRAM leaving enough VRAM which can be used to store conversational history/context in RAG.
5GB of RAM preferred while running.    

Ollama is better than manually optimizing inference by using 'bitsandbytes' and 'accelerate' libraries.
Ollama automatically utilizes GPU resources if available.

The chunking strategy is set to "hi_res" which will take a longer time to chunk. If speed is important then change the strategy to "fast".