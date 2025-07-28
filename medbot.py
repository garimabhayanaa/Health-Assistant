import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from image_processing import analyse_image_with_query, encode_image

HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt():
    return PromptTemplate(template="""
        Use the context to answer the user's question.
        If you don’t know, say so. Don't make up an answer.
        Context: {context}
        Question: {question}
    """, input_variables=["context", "question"])

def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.5,
        max_new_tokens=512,  
        huggingfacehub_api_token=os.environ["HF_TOKEN"]
    )

def main():
    # Page configuration with custom theme and favicon
    st.set_page_config(
        page_title="Health Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    vectorstore = get_vectorstore()
    qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )
    
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
        * Make the chat input border and text orange */
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
        st.chat_message(message['role']).markdown(message['content'])

    # Chat Input
    prompt = st.chat_input("💬 Type your message...")
    if prompt:
        st.chat_message("User").markdown(f"**User:** {prompt}")
        st.session_state.messages.append({'role': 'User', 'content': prompt})
        response = qa_chain.invoke({'query': prompt})
        result = response["result"]
        st.chat_message("Assistant").markdown(result)
        st.session_state.messages.append({'role': 'Assistant', 'content': result})
    st.write("---")
    st.session_state.uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    # Image Handling
    if st.session_state.uploaded_image:
        st.session_state.image_query = st.text_input("Enter your query for image analysis:", value=st.session_state.image_query)
        if st.button("Analyze Image"):
            encoded_image = encode_image(st.session_state.uploaded_image)
            result = analyse_image_with_query(st.session_state.image_query, encoded_image)
            # Append user query and response to chat history
            st.chat_message("User").markdown(f"**User (Image Query):** {st.session_state.image_query}")
            st.session_state.messages.append({'role': 'User', 'content': st.session_state.image_query})
            st.chat_message("Assistant").markdown(result)
            st.session_state.messages.append({'role': 'Assistant', 'content': result})

            # Properly clear session state variables before rerun
            del st.session_state["uploaded_image"]
            del st.session_state["image_query"]
            st.rerun()

if __name__ == "__main__":
    main()
