import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# Try to import GROQ, fall back to HuggingFace if not available
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("langchain_groq not found. Install with: pip install langchain-groq")

# Fallback to HuggingFace if GROQ not available
if not GROQ_AVAILABLE:
    from langchain_huggingface import HuggingFaceEndpoint

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
    """Load LLM with GROQ as primary, HuggingFace as fallback"""    
    # Try GROQ first if available and API key is set
    if GROQ_AVAILABLE and GROQ_API_KEY:
        try:
            return ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.3-70b-versatile",  # Updated to current model
                temperature=0.4,
                max_tokens=400
            )
        except Exception as e:
            st.warning(f"GROQ API failed: {str(e)}. Falling back to HuggingFace...")
    
    # Fallback to HuggingFace
    if not HF_TOKEN:
        st.error("Neither GROQ_API_KEY nor HF_TOKEN is set. Please set at least one API key.")
        return None
    
    try:
        # Try multiple models in order of preference
        models_to_try = [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "microsoft/DialoGPT-medium",
            "google/flan-t5-large"
        ]
        
        for model in models_to_try:
            try:
                task = "text2text-generation" if "flan-t5" in model else "text-generation"
                return HuggingFaceEndpoint(
                    repo_id=model,
                    task=task,
                    temperature=0.4,
                    max_new_tokens=400,
                    huggingfacehub_api_token=HF_TOKEN
                )
            except Exception as e:
                st.warning(f"Model {model} failed: {str(e)}")
                continue
        
        st.error("All models failed to load")
        return None
        
    except Exception as e:
        st.error(f"Error loading HuggingFace models: {str(e)}")
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Health Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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

    st.title("AI-Powered Health Assistant")

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
    prompt = st.chat_input("💬 Ask me anything about health...")
    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
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
            error_msg = f"❌ Error generating response: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

    st.markdown("---")
    
    # Image Analysis Section
    st.markdown("### Image Analysis")
    uploaded_file = st.file_uploader(
        "Upload a medical image for analysis", 
        type=["png", "jpg", "jpeg"],
        help="Upload X-rays, scans, or other medical images"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            image_query = st.text_area(
                "What would you like to know about this image?", 
                value=st.session_state.get("image_query", ""),
                placeholder="Describe what you see in this image..."
            )
            
            if st.button("🔍 Analyze Image", use_container_width=True):
                if not image_query.strip():
                    st.warning("Please enter a question about the image.")
                else:
                    try:
                        with st.spinner("🔍 Analyzing image..."):
                            encoded_image = encode_image(uploaded_file)
                            result = analyse_image_with_query(image_query, encoded_image)
                        
                        # Add to chat history
                        with st.chat_message("user"):
                            st.markdown(f"**Image Analysis Query:** {image_query}")
                        st.session_state.messages.append({
                            'role': 'user', 
                            'content': f"Image Analysis: {image_query}"
                        })
                        
                        with st.chat_message("assistant"):
                            st.markdown(result)
                        st.session_state.messages.append({
                            'role': 'assistant', 
                            'content': result
                        })
                        
                        # Reset image query
                        st.session_state.image_query = ""
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = f"❌ Error analyzing image: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            'role': 'assistant', 
                            'content': error_msg
                        })

    # Footer with instructions
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ask specific health questions for better responses
    - Upload medical images for detailed analysis
    - The AI uses medical literature to provide informed answers
    - **Disclaimer**: This is for informational purposes only. Always consult healthcare professionals.
    """)

if __name__ == "__main__":
    main()
