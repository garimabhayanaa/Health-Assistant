import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from image_processing import analyse_image_with_query, encode_image

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"

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
        if GROQ_API_KEY:
            # Use GROQ API (faster and more reliable)
            return ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="mixtral-8x7b-32768",  # or "llama2-70b-4096"
                temperature=0.4,
                max_tokens=400
            )
        else:
            st.error("GROQ_API_KEY is not set. Please set your GROQ API key.")
            return None
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Health Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Debug info
    st.sidebar.write(f"**Debug Info:**")
    st.sidebar.write(f"GROQ API Set: {'Yes' if GROQ_API_KEY else 'No'}")
    st.sidebar.write(f"HF Token Set: {'Yes' if HF_TOKEN else 'No'}")
    
    # Check API keys
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY environment variable is not set. Please set your GROQ API key.")
        st.info("Get your GROQ API key from: https://console.groq.com/keys")
        return
    
    # Load components
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return
    
    llm = load_llm()
    if llm is None:
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
        div[data-testid="stChatInput"] textarea {
            border: 1px solid black !important;
            color: #77DD77 !important;
        }
        div[data-testid="stFileUploader"] {
            color: #77DD77 !important;
        }
        div[data-testid="stFileUploader"] label {
            color: #77DD77 !important;
        }
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
    st.success("✅ Using GROQ API for faster responses!")

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
        
        # Get response
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
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        image_query = st.text_input("Enter your query for image analysis:", 
                                   value=st.session_state.get("image_query", ""))
        
        if st.button("Analyze Image") and image_query:
            try:
                with st.spinner("Analyzing image..."):
                    encoded_image = encode_image(uploaded_file)
                    result = analyse_image_with_query(image_query, encoded_image)
                
                with st.chat_message("user"):
                    st.markdown(f"**User (Image Query):** {image_query}")
                st.session_state.messages.append({'role': 'user', 'content': f"Image Query: {image_query}"})
                
                with st.chat_message("assistant"):
                    st.markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})
                
                st.session_state.image_query = ""
                st.rerun()
                
            except Exception as e:
                error_msg = f"Error analyzing image: {str(e)}"
                st.error(error_msg)

if __name__ == "__main__":
    main()
