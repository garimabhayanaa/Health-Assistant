# AI-Powered Medical Assistant — Product Context

## Overview
The AI-Powered Medical Assistant is a prototype healthcare support tool designed to explore how Retrieval-Augmented Generation (RAG) and multimodal LLMs can be used responsibly to provide medical information. The focus of the product was not diagnosis, but accuracy, grounding, and trust in a sensitive, high-risk domain.

## Problem & Context
Medical AI systems face a fundamental challenge:
users often expect authoritative answers, while AI systems are probabilistic and error-prone.

The core problem addressed was:
How can an AI assistant provide useful medical information without overstepping into unsafe or misleading medical advice?
This required prioritizing factual grounding, transparency, and constraint design over feature breadth.

## Users & Assumptions

### Intended users
1. Individuals seeking preliminary medical information
2. Users exploring medical images or reports for educational purposes

### Key assumptions
1. Users value reliable context more than confident answers
2. Grounded responses reduce over-trust in AI systems
3. Clear limitations are essential in healthcare applications

## Solution & Key Decisions
Several product decisions were made to align with these assumptions:
1. Retrieval-Augmented Generation (RAG)
Chosen to reduce hallucinations by grounding responses in verified medical sources rather than relying solely on model inference.
2. Multimodal input support (text + image)
Included to reflect real-world medical queries while carefully limiting interpretive claims.
3. Prototype positioning
The system was explicitly framed as an informational assistant, not a diagnostic tool.

## Tradeoffs & Constraints
Key tradeoffs shaped the system:
1. Coverage vs safety
The assistant limits the scope of responses to avoid diagnostic certainty.
2. Model capability vs explainability
Preference was given to responses that could be traced back to retrieved knowledge.
3. Speed vs reliability
Retrieval steps added latency but improved trustworthiness.
These constraints intentionally reduced system ambition in favor of safer behavior.

## Learnings & Reflection
Building in a healthcare context highlighted that:
1. Technical capability must be constrained by domain risk
2. Product success is defined as much by what the system refuses to do as by what it can do
3. Trust and user expectations are as important as accuracy

## Improvements
With broader testing and regulatory guidance:
1. Add clearer confidence indicators and disclaimers
2. Expand multilingual access while preserving grounding
3. Introduce structured evaluation for response safety

## Technical Details 

### 🔹 Key Features  
- **AI-Powered Medical Insights**: Supports both **text and image-based** medical queries.  
- **Retrieval-Augmented Generation (RAG)**: Enhances accuracy by retrieving relevant medical knowledge before generating responses.  
- **Multimodal LLM Integration**: Uses **Groq’s advanced LLM** to analyze both textual and visual medical data.  
- **Efficient Search with FAISS**: Enables **fast and accurate** information retrieval from medical datasets.  
- **Seamless User Experience**: Designed for accessibility, ensuring **instant AI-driven medical support**.  

### Technologies Used  
- **Vector Search**: FAISS (Facebook AI Similarity Search)  
- **AI Models**: Hugging Face NLP & Groq’s multimodal LLM  
- **Text & Image Processing**: OpenCV, PIL, NumPy  
- **Backend**: Python (FastAPI)  
- **Frontend**: Streamlit for an interactive user experience  

### Installation  

#### Prerequisites  
- **Python 3.8+** installed  
- Hugging Face API key  
- Groq API key  

#### Steps  
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/garimabhayanaa/Health-Assistant
   cd Health-Assistant
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

### Usage
1. Enter a medical query or upload a medical image.
2. AI processes the input using NLP and image analysis.
3. Receive an accurate medical response powered by RAG and multimodal AI.

## About the Project
This project was developed as part of the AICTE TechSaksham Initiative (by Microsoft & SAP), where I worked as an AI Intern focusing on AI-driven healthcare solutions. The experience strengthened my skills in retrieval-based AI, multimodal learning, and scalable AI development.

## Acknowledgments
Special thanks to Microsoft, SAP, and AICTE TechSaksham for their mentorship and resources.
