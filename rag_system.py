import os
import tempfile
import numpy as np
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pydantic import SecretStr
from typing import Optional, Dict, List


class RAGSystem:

    def __init__(self):
        """
        Initialize the RAG system with FAISS vector store and OpenAI integration.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        self.has_api_key = api_key is not None
        self.documents = []
        self.document_sources = {}  # Map document chunks to their source filenames
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len)
        self.vector_store = None
        self.embeddings = None

        # Only initialize embeddings if API key is available
        if self.has_api_key:
            try:
                self.embeddings = OpenAIEmbeddings()
                # Initialize an empty vector store
                self._initialize_empty_vector_store()
            except Exception as e:
                print(f"Failed to initialize OpenAI embeddings: {str(e)}")
        else:
            print(
                "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable."
            )

    def _initialize_empty_vector_store(self):
        """
        Initialize an empty FAISS vector store.
        """
        if self.embeddings is None:
            return

        try:
            # Create an empty document to initialize the vector store
            self.vector_store = FAISS.from_documents(
                [Document(page_content="", metadata={})], self.embeddings)
        except Exception as e:
            print(f"Failed to initialize vector store: {str(e)}")

    def add_document(self, file_path, original_filename=None):
        """
        Process a document and add it to the RAG system.
        
        Args:
            file_path (str): Path to the document file
            original_filename (str, optional): Original filename for citation purposes
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.has_api_key or self.embeddings is None:
            raise Exception(
                "OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable."
            )

        try:
            # Get the source name for citation (use original filename if provided)
            source_name = original_filename if original_filename else os.path.basename(file_path)
            tmp_csv_path = None
            
            # Determine file type and load accordingly
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            elif file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path)
            elif file_path.endswith('.xlsx'):
                # For Excel files, we convert to CSV first
                df = pd.read_excel(file_path)
                # Create a temporary CSV file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_csv_path = tmp_file.name
                df.to_csv(tmp_csv_path, index=False)
                loader = CSVLoader(tmp_csv_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")

            # Load and split the document
            loaded_documents = loader.load()
            self.documents.extend(loaded_documents)

            # Split the document into chunks
            split_docs = self.text_splitter.split_documents(loaded_documents)
            
            # Add source information to each document chunk
            for doc in split_docs:
                doc.metadata["source"] = source_name

            # Add to vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    split_docs, self.embeddings)
            else:
                # Add to existing vector store
                self.vector_store.add_documents(split_docs)
                
            # Clean up temporary file if using Excel
            if tmp_csv_path and os.path.exists(tmp_csv_path):
                try:
                    os.unlink(tmp_csv_path)
                except Exception as e:
                    print(f"Error cleaning up temporary file: {e}")

            return True

        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def generate_response(self, query, chat_history, temperature=0.2):
        """
        Generate a response to a user query using RAG.
        
        Args:
            query (str): User query
            chat_history (list): List of previous conversation messages
            temperature (float): Temperature parameter for the LLM
            
        Returns:
            str: Generated response
        """
        if not self.has_api_key:
            return "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."

        if self.vector_store is None or not self.documents:
            return "No documents have been loaded into the knowledge base yet."

        try:
            # Create the LLM
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            llm = ChatOpenAI(model="gpt-4o", temperature=temperature)

            # Create an optimized retriever from the vector store
            retriever = self.vector_store.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": 3,  # Reduced to 3 most relevant documents for faster retrieval
                    "fetch_k": 5  # Fetch 5 documents first, then pick 3 most diverse
                }
            )

            # Create a template that includes conversation history and retrieved context
            template = """
            You are a helpful AI assistant. Use the following context and conversation history to answer the user's query.
            If you don't know the answer based on the provided context, say so instead of making up information.
            
            Context:
            {context}
            
            Conversation History:
            {chat_history}
            
            User Query: {question}
            
            Please provide a detailed and helpful answer:
            """

            # Format chat history for the prompt
            chat_history_text = ""
            # Take just the last 6 messages for context if available
            recent_messages = chat_history[-6:] if len(chat_history) >= 6 else chat_history
            try:
                for message in recent_messages:
                    if isinstance(message, dict) and "role" in message and "content" in message:
                        role = message["role"]
                        content = message["content"]
                        chat_history_text += f"{role}: {content}\n"
            except Exception as e:
                print(f"Error formatting chat history: {str(e)}")
                # Provide a simple fallback if there are format issues
                chat_history_text = "No previous conversation available."

            # Create prompt template
            prompt = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=template)

            # Create a chain that includes source documents in the return value
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        input_variables=["context", "question"],
                        template="""
                        You are a helpful AI assistant. Use the following context to answer the user's question.
                        If you don't know the answer based on the provided context, say so instead of making up information.
                        
                        Context:
                        {context}
                        
                        User Question: {question}
                        
                        Please provide a detailed and helpful answer:
                        """
                    ),
                    "verbose": False
                })

            # Generate response with simplified inputs
            response = qa_chain.invoke({
                "query": query,
                "question": query
            })
            
            # Extract the response text and source documents
            result = response["result"]
            source_documents = response.get("source_documents", [])
            
            # Extract unique source document names
            cited_sources = set()
            for doc in source_documents:
                if "source" in doc.metadata:
                    cited_sources.add(doc.metadata["source"])
            
            # Add source citations to the response
            if cited_sources:
                sources_text = ", ".join(sorted(list(cited_sources)))
                result += f"\n\nSource: {sources_text}"
            
            return result

        except Exception as e:
            return f"An error occurred while generating a response: {str(e)}"
