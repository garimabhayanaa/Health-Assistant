import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from image_processing import analyse_image_with_query, encode_image

# Fix for torch classes path error
import torch
import sys

HF_TOKEN = os.getenv("HF_TOKEN")
# Use verified working Mistral models
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Most stable Mistral model
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def set_custom_prompt():
    return PromptTemplate(template="""
        Use the context to answer the user's question.
        If you don't know, say so. Don't make up an answer.
        Context: {context}
        Question: {question}
        
        Answer:
    """, input_variables=["context", "question"])

def load_llm():
    try:
        # Primary: Try with Mistral-7B-Instruct-v0.3 (verified working)
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            temperature=0.4,
            max_new_tokens=400,
            huggingfacehub_api_token=HF_TOKEN
        )
    except Exception as e:
        st.warning(f"Primary model failed: {str(e)}")
        try:
            # Fallback 1: Mistral v0.2
            return HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                task="text-generation",
                temperature=0.4,
                max_new_tokens=400,
                huggingfacehub_api_token=HF_TOKEN
            )
        except Exception as e2:
            st.warning(f"Fallback 1 failed: {str(e2)}")
            try:
                # Fallback 2: Google Flan-T5 (very reliable)
                return HuggingFaceEndpoint(
                    repo_id="google/flan-t5-large",
                    task="text2text-generation",
                    temperature=0.4,
                    max_new_tokens=400,
                    huggingfacehub_api_token=HF_TOKEN
                )
            except Exception as e3:
                st.error(f"All models failed. Last error: {str(e3)}")
                return None

def main():
    # Page configuration with custom theme and favicon
    st.set_page_config(
        page_title="Health Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Debug: Show which model we're trying to use
    st.sidebar.write(f"**Debug Info:**")
    st.sidebar.write(f"Primary Model: mistralai/Mistral-7B-Instruct-v0.3")
    st.sidebar.write(f"HF Token Set: {'Yes' if HF_TOKEN else 'No'}")
    
    # Check if required environment variables are set
    if not HF_TOKEN:
        st.error("HF_TOKEN environment variable is not set. Please set your Hugging Face token.")
        return
    
    # Load components with error handling
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Failed to load vector store. Please check your vector store path.")
        return
    
    llm = load_llm()
    if llm is None:
        st.error("Failed to load language model. Please check your Hugging Face token and model availability.")
        return
    
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return
    
    # Page Styling
    st.markdown("""
    <style>
        body, .stApp {
            background-color: white !important;
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6, p, label , div {
            color: black !important;
        }
        .stTextInput, .stFileUploader {
            background-color: white !important;
            color: black !important;
            border: 1px solid black !important;
        }
        .stButton>button {
            background-color: #77DD77 !important;
            color : white !important;
            border: 1px solid black !important;
        }
        hr {
            border: 1px solid black !important;
            opacity: 0.2;
        }
        .stMarkdown {
            color: black !important;  
        }
        /* Make the chat input border and text orange */
        div[data-testid="stChatInput"] textarea {
            border: 1px solid black !important;
            color: #77DD77 !important;
        }
        /* Make the file uploader button orange */
        div[data-testid="stFileUploader"] {
            color: #77DD77 !important;
        }
        /* Change file uploader text color */
        div[data-testid="stFileUploader"] label {
            color: #77DD77 !important;
        }
        
        /* Change file uploader button color */
        div[data-testid="stFileUploader"] button {
            background-color: #77DD77 !important;
            color: black !important;
            border-radius: 8px !important;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Welcome to your AI-Powered Health Assistant!")

    # Session State Initialization
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "image_query" not in st.session_state:
        st.session_state.image_query = ""

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat Input
    prompt = st.chat_input("💬 Type your message...")
    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"**User:** {prompt}")
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Get response with error handling
        try:
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

    st.write("---")
    
    # Image Handling
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file
        
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        image_query = st.text_input("Enter your query for image analysis:", 
                                   value=st.session_state.get("image_query", ""))
        
        if st.button("Analyze Image") and image_query:
            try:
                with st.spinner("Analyzing image..."):
                    encoded_image = encode_image(uploaded_file)
                    result = analyse_image_with_query(image_query, encoded_image)
                
                # Display user query and response
                with st.chat_message("user"):
                    st.markdown(f"**User (Image Query):** {image_query}")
                st.session_state.messages.append({'role': 'user', 'content': f"Image Query: {image_query}"})
                
                with st.chat_message("assistant"):
                    st.markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})
                
                # Clear the image query after processing
                st.session_state.image_query = ""
                st.rerun()
                
            except Exception as e:
                error_msg = f"Error analyzing image: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

if __name__ == "__main__":
    main()
