# RAG Chatbot - System Overview

## Overview

This is boilerplate code for a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that allows users to upload documents (PDF, TXT, DOCX) and chat with an AI about the content of those documents. The system uses FAISS for efficient vector search and OpenAI for embeddings and chat completion.

## System Architecture

The application follows a modular architecture with the following key components:

1. **Web Interface (Streamlit)**: Provides a user-friendly interface for document uploading and chat interaction.
2. **RAG System**: Core component that handles document processing, vector storage, and query generation.
3. **Utility Functions**: Helper functions for generating responses and managing conversation state.

The system uses a Retrieval-Augmented Generation (RAG) approach, which enhances Large Language Model (LLM) responses by retrieving relevant context from user-uploaded documents before generating answers.

## Key Components

### 1. Streamlit Application (app.py)
- Serves as the main entry point and UI layer
- Handles document uploading via file uploader
- Manages session state for conversation history
- Renders chat interface using streamlit-chat components

### 2. RAG System (rag_system.py)
- Implements the core RAG functionality
- Uses FAISS for vector storage and similarity search
- Processes documents with LangChain document loaders
- Creates embeddings using OpenAI's embedding models
- Splits documents into chunks for efficient retrieval
- Manages the retrieval and generation pipeline

### 3. Utility Functions (utils.py)
- Provides helper functions for response generation
- Handles conversation management
- Manages error handling for API interactions
- Provides functionality to reset conversations

## Data Flow

1. **Document Processing**:
   - User uploads documents through the Streamlit interface
   - Documents are temporarily saved to disk
   - RAG system processes documents (loading, splitting, embedding)
   - Document chunks are stored in the FAISS vector store

2. **Query Processing**:
   - User enters a question in the chat interface
   - Query is processed to retrieve relevant document chunks from FAISS
   - Retrieved content provides context to the LLM
   - Response is generated and displayed to the user

3. **Conversation Management**:
   - Chat history is maintained in session state
   - Previous interactions can influence future responses
   - User can reset the conversation when needed

## External Dependencies

### Core Libraries
- **Streamlit & streamlit-chat**: For the web interface and chat components
- **LangChain**: Provides the framework for document loading, chunking, and retrieval
- **OpenAI**: Used for embeddings and chat completion
- **FAISS**: Vector database for efficient similarity search

### Document Processing
- **PyPDF2**: For processing PDF documents
- **python-docx**: For processing DOCX files

### Environment Requirements
- Requires an OpenAI API key set as an environment variable (`OPENAI_API_KEY`)
- Python 3.11 or higher

## Deployment Strategy

1. **Dependencies**:
   - Dependencies are managed through `pyproject.toml`
   - Specific version requirements ensure compatibility

2. **Environment Variables**:
   - The OpenAI API key must be set as an environment variable

## Implementation Notes

- The system initializes an empty FAISS vector store on startup, which gets populated as documents are added.
- The RAG system could be extended to support more document types or different embedding models.
- The current implementation uses OpenAI's models, but the architecture could be adapted to use open-source alternatives.

## Future Improvements

- Complete implementation of the RAG system methods (document processing, retrieval)
- Add support for more document formats
- Implement document metadata management
- Add user authentication for personalized document collections
- Optimize chunking strategies for better retrieval performance
- Add caching mechanisms to improve response time for repeated queries