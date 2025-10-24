import os

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from langchain_core.documents import Document

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough    

def make_dirs():
    os.makedirs("./static/extracted_images", exist_ok=True)

# A utility function to format documents for the prompt context
def format_docs(docs):
    """Formats the retrieved documents into a single string for the LLM context."""
    return "\n\n".join(doc.page_content for doc in docs)

def load_and_chunk(file_path):
    # Load and partition the PDF
    elements = partition_pdf(
        file_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_output_dir="./static/extracted_images",
        infer_table_structure=True,
    )
    elements_to_json(elements, filename="./static/elements.json")

    with open("./static/elements.json", "r") as f:
        json_elements = json.load(f)
    
    raw_text = ""
    for element in json_elements:
        if element.get("text"):
            raw_text += element["text"] + "\n\n"
        
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_text)

    # Create Document objects with metadata
    documents = [
        Document(
            page_content=chunk, 
            metadata={"source": os.path.basename(file_path)}
        )
        for chunk in chunks
    ]
    return documents

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    return vector_store

def load_llm():
    llm = ChatOllama(model="mistral:7b")
    return llm

def build_rag_chain(vector_store, llm):
    # 1. Define the Retriever (top 3 most relevant documents)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Define the Prompt Template
    template = """You are a helpful assistant specialized in analyzing documents.
    Answer the user's question based on the following context.
    If you cannot find the answer in the context and you know the CORRECT answer then answer it.
    Try to give the reference source/links/page from where you got the answer.
    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Build the RAG Chain using LCEL
    rag_chain = (
        # RunnablePassthrough ensures the 'question' input is passed directly
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain