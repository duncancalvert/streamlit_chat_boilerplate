import os
import streamlit as st
from streamlit_chat import message
import tempfile
from dotenv import load_dotenv

# Custom libs
from utils import generate_response, reset_conversation
from rag_system import RAGSystem

load_dotenv()


# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "api_key_error" not in st.session_state:
    st.session_state.api_key_error = False
if "folder_loaded" not in st.session_state:
    st.session_state.folder_loaded = False
    
# Initialize RAG system
if "rag_system" not in st.session_state:
    try:
        st.session_state.rag_system = RAGSystem()
        if os.getenv("OPENAI_API_KEY") is None:
            st.session_state.api_key_error = True
            
        # Check for default document folder on startup
        default_docs_folder = os.getenv("DOCUMENTS_FOLDER")
        if default_docs_folder and os.path.isdir(default_docs_folder) and not st.session_state.folder_loaded:
            with st.spinner(f"Loading documents from {default_docs_folder}..."):
                try:
                    successful, total = st.session_state.rag_system.process_folder(default_docs_folder)
                    if successful > 0:
                        st.session_state.document_uploaded = True
                        st.session_state.folder_loaded = True
                        st.success(f"Successfully loaded {successful} documents from the default folder.")
                except Exception as e:
                    st.error(f"Error loading documents from default folder: {str(e)}")
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        st.session_state.api_key_error = True

# App title and description
st.title("RAG Chatbot with FAISS")
st.subheader("Chat with your documents using AI")

# Sidebar for document upload and system settings
with st.sidebar:
    st.header("Document Upload")
    st.write("Upload documents to create a knowledge base for the chatbot.")
    
    # Option to process files from a folder
    folder_path = st.text_input("Enter folder path containing documents (optional)", key="folder_path")
    
    if folder_path and st.button("Process Folder"):
        with st.spinner("Processing documents from folder..."):
            try:
                successful, total = st.session_state.rag_system.process_folder(folder_path)
                if successful > 0:
                    st.session_state.document_uploaded = True
                    st.success(f"Successfully processed {successful} out of {total} files from the folder.")
                else:
                    st.warning(f"No compatible documents found in {folder_path}. Please check the path and try again.")
            except Exception as e:
                st.error(f"Error processing folder: {str(e)}")
    
    # Or upload individual files
    st.markdown("### Or upload individual files")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX, XLSX, or CSV files", 
        type=["pdf", "txt", "docx", "xlsx", "csv"], 
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Keep track of processed files to avoid re-processing
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            newly_processed = False
            
            for uploaded_file in uploaded_files:
                # Check if this file has already been processed (using name as identifier)
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                
                if file_id not in st.session_state.processed_files:
                    # Save the uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process the document
                    try:
                        # Pass both the temporary path and the original filename
                        st.session_state.rag_system.add_document(
                            file_path=tmp_path, 
                            original_filename=uploaded_file.name
                        )
                        st.session_state.document_uploaded = True
                        st.session_state.processed_files.add(file_id)
                        newly_processed = True
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
            
            if newly_processed:
                st.success("Documents processed successfully!")
    
    st.header("Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Reset button
    if st.button("Reset Conversation"):
        reset_conversation()
        st.success("Conversation has been reset")
        st.rerun()

# Show API key warning if needed
if st.session_state.api_key_error:
    st.warning("⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable to use this app.")

# Display chat messages
for i, (role, content) in enumerate(st.session_state.messages):
    if role == "user":
        message(content, is_user=True, key=f"user_{i}")
    else:
        message(content, is_user=False, key=f"assistant_{i}")

# Chat input
user_input = st.chat_input("Type your message here...")

# Handle user input
if user_input:
    # Add user message to chat history
    st.session_state.messages.append(("user", user_input))
    
    # Check if OpenAI API key is set
    if st.session_state.api_key_error:
        st.session_state.messages.append(("assistant", "Please set the OPENAI_API_KEY environment variable to use this app."))
        st.rerun()
    
    # Check if documents have been uploaded
    if not st.session_state.document_uploaded:
        st.session_state.messages.append(("assistant", "Please upload documents to the knowledge base first."))
        st.rerun()
    
    # Generate response
    with st.spinner("Thinking..."):
        try:
            response = generate_response(
                user_input, 
                st.session_state.rag_system, 
                st.session_state.messages, 
                temperature
            )
            # Add response to chat history
            st.session_state.messages.append(("assistant", response))
        except Exception as e:
            st.session_state.messages.append(("assistant", f"Error: {str(e)}"))
    
    # Rerun the app to update the chat
    st.rerun()