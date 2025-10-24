import streamlit as st
import tempfile
import os

from rag import load_and_chunk, create_vector_store, load_llm, build_rag_chain

from langchain_core.messages import HumanMessage, AIMessage

def main():
    st.set_page_config(page_title="Local RAG PDF Chatbot")
    st.title("📄 Local RAG PDF Chatbot")
    st.info("This app uses LangChain, FAISS, and a local Mistral-7B model (via Ollama) to answer questions based on your uploaded PDFs.")

    # --- 1. Initialize Session State ---
    # Used to persist the RAG chain and chat history across Streamlit reruns
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "messages" not in st.session_state:
        # Chat history is stored as a list of dictionaries
        st.session_state.messages = []
        
    # --- 2. Sidebar for File Upload and Processing ---
    with st.sidebar:
        st.subheader("Upload PDF Documents")
        files = st.file_uploader(
            "Upload one or more PDF documents:", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        if st.button("Process Documents") and files:
            # Clear previous chat history when new files are processed
            st.session_state.messages = []
            
            with st.spinner("Processing documents... This may take a moment."):
                file_paths = []
                all_documents = []
                
                # 1. Save files temporarily and read contents
                for uploaded_file in files:
                    # Use tempfile to save the file
                    # This ensures LangChain's PDF reader can access it from disk
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_path = temp_file.name
                        file_paths.append(temp_path)
                    
                    # 2. Load, split, and chunk documents
                    documents = load_and_chunk(temp_path)
                    all_documents.extend(documents)
                
                if all_documents:
                    # 3. Create Vector Store from ALL aggregated documents
                    vectorstore = create_vector_store(all_documents)
                    
                    # 4. Load LLM and Build RAG Chain
                    llm = load_llm()
                    qa_chain = build_rag_chain(vectorstore, llm)
                    
                    # 5. Store the chain in session state
                    st.session_state.qa_chain = qa_chain
                    st.success(f"Successfully processed {len(files)} file(s) with {len(all_documents)} chunks. You can now ask questions!")
                else:
                    st.error("Could not extract any text or documents from the uploaded PDFs.")

                # Clean up temporary files 
                for path in file_paths:
                    if os.path.exists(path):
                        os.unlink(path)

    # --- 3. Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- 4. Main Chat Input and Response Generation ---
    if st.session_state.qa_chain is not None:
        if prompt := st.chat_input("Ask a question about your documents..."):
            
            # Add user message to chat history and display it
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Call the RAG chain and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant context and generating response..."):
                    try:
                        # Invoke the chain with the user's prompt
                        response = st.session_state.qa_chain.invoke(prompt)
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred during chain invocation. Ensure Ollama is running and 'mistral:7b' is pulled. Error: {e}")

    else:
        st.warning("Please upload and process your PDF documents in the sidebar to start chatting.")

if __name__ == "__main__":
    main()
