import os

# Setup GROQ API KEY
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")

# Convert image to required format
import base64

def encode_image(uploaded_file):
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()  # Read file contents as bytes
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        return encoded_image
    return None


# Setup multimodal LLM
from groq import Groq
query= "placeholder query"

def analyse_image_with_query(query,encoded_image):
    client= Groq()
    query= query
    model="llava-1.6-34b"
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type":"text",
                    "text": "query",
                }, 
                {
                    "type":"image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                }
            ]
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return(chat_completion.choices[0].message.content)
