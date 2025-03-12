# AI-Powered Medical Assistant  

## Overview  
The **AI-Powered Medical Assistant** is an advanced AI-driven healthcare solution that utilizes **Retrieval-Augmented Generation (RAG)** and **multimodal LLMs** to provide precise **medical insights**. Users can **submit text-based queries** or **upload medical images** for AI-driven analysis, enhancing **diagnostic accuracy** and medical decision-making.  

### 🔹 Key Features  
- **AI-Powered Medical Insights**: Supports both **text and image-based** medical queries.  
- **Retrieval-Augmented Generation (RAG)**: Enhances accuracy by retrieving relevant medical knowledge before generating responses.  
- **Multimodal LLM Integration**: Uses **Groq’s advanced LLM** to analyze both textual and visual medical data.  
- **Efficient Search with FAISS**: Enables **fast and accurate** information retrieval from medical datasets.  
- **Seamless User Experience**: Designed for accessibility, ensuring **instant AI-driven medical support**.  

## Technologies Used  
- **Vector Search**: FAISS (Facebook AI Similarity Search)  
- **AI Models**: Hugging Face NLP & Groq’s multimodal LLM  
- **Text & Image Processing**: OpenCV, PIL, NumPy  
- **Backend**: Python (FastAPI)  
- **Frontend**: Streamlit for an interactive user experience  

## Installation  

### Prerequisites  
- **Python 3.8+** installed  
- Hugging Face API key  
- Groq API key  

### Steps  
1. **Clone the Repository:**  
   ```bash
   git clone <repository-url>
   cd AI-Medical-Assistant
2. **Install Dependencies:** 
    ```bash
    pip install -r requirements.txt
3. **Set Up Environment Variables:**
    Create a .env file and add:
      HUGGINGFACE_API_KEY=<your-key>
      GROQ_API_KEY=<your-key>
4. **Run the Application:**
    ```bash
    python app.py
5. **Access the Assistant:**
    Open your browser and visit http://localhost:8501 to use the AI assistant.

## Usage
1. Enter a medical query or upload a medical image.
2. AI processes the input using NLP and image analysis.
3. Receive an accurate medical response powered by RAG and multimodal AI.

## Future Enhancements
- Integration with Medical Databases
- Multi-Language Support
- Mobile App Version
## About the Project
This project was developed as part of the AICTE TechSaksham Initiative (by Microsoft & SAP), where I worked as an AI Intern focusing on AI-driven healthcare solutions. The experience strengthened my skills in retrieval-based AI, multimodal learning, and scalable AI development.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to Microsoft, SAP, and AICTE TechSaksham for their mentorship and resources.
