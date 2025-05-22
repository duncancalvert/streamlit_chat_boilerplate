import os
import tempfile
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pydantic import SecretStr
from typing import Optional


class RAGSystem:
    def __init__(self):
        """
        Initialize the RAG system with FAISS vector store and OpenAI integration.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        self.has_api_key = api_key is not None
        self.documents = []
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

    def add_document(self, file_path):
        """
        Process a document and add it to the RAG system.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.has_api_key or self.embeddings is None:
            raise Exception(
                "OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable."
            )

        try:
            # Determine file type and load accordingly
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            elif file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")

            # Load and split the document
            loaded_documents = loader.load()
            self.documents.extend(loaded_documents)

            # Split the document into chunks
            split_docs = self.text_splitter.split_documents(loaded_documents)

            # Add to vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    split_docs, self.embeddings)
            else:
                # Add to existing vector store
                self.vector_store.add_documents(split_docs)

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

            # Create a retriever from the vector store
            retriever = self.vector_store.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={"k":
                               5}  # Retrieve top 5 most relevant documents
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
            for message in chat_history[
                    -6:]:  # Include last 6 messages for context
                role = message["role"]
                content = message["content"]
                chat_history_text += f"{role}: {content}\n"

            # Create prompt template
            prompt = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=template)

            # Create a chain to retrieve documents and generate an answer
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={
                    "prompt": prompt,
                    "verbose": False
                })

            # Generate response
            response = qa_chain({
                "query": query,
                "chat_history": chat_history_text
            })

            return response["result"]

        except Exception as e:
            return f"An error occurred while generating a response: {str(e)}"
