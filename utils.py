import os
import streamlit as st

def generate_response(user_input, rag_system, message_history, temperature=0.7):
    """
    Generate a response using the RAG system.
    
    Args:
        user_input (str): The user's input message
        rag_system (RAGSystem): The initialized RAG system
        message_history (list): Previous messages in the conversation
        temperature (float): Temperature parameter for the LLM
        
    Returns:
        str: The generated response
    """
    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        
        # Create chat history context for the LLM
        chat_history = []
        for msg in message_history:
            role, content = msg
            # Convert 'user' and 'assistant' to OpenAI's expected format
            if role == "user":
                chat_history.append({"role": "user", "content": content})
            elif role == "assistant":
                chat_history.append({"role": "assistant", "content": content})
        
        # Generate response with RAG
        response = rag_system.generate_response(user_input, chat_history, temperature)
        return response
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

def reset_conversation():
    """
    Reset the conversation by clearing message history and maintaining the RAG system.
    """
    # Clear message history
    st.session_state.messages = []
