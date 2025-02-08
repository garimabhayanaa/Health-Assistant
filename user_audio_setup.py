# Setup audio recorder
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s') 

def record_audio(file_path="recorded_audio.mp3", timeout=20, phrase_time_limit=None):
    """
    Records audio from the microphone and saves it as an MP3 file.
    
    Args:
        file_path (str): Path to save the recorded audio.
        timeout (int): Maximum time to wait for a phrase to start (in seconds).
        phrase_time_limit (int): Maximum time to wait for a phrase to finish (in seconds).
    
    Returns:
        str: File path of the recorded audio.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Convert recorded audio to MP3
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format='mp3', bitrate='128K')
            logging.info(f"Audio saved to {file_path}")
            return file_path
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None

# Setup speech to text-STT-model for transcription
import os
from groq import Groq

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

def transcribe_with_groq(audio_filepath,GROQ_API_KEY):
    client= Groq(api_key=GROQ_API_KEY)
    stt_model="whisper-large-v3"
    audio_file= open(audio_filepath,"rb")
    transcription= client.audio.transcriptions.create(
        model= stt_model,
        file = audio_file,
        language= "en"
    )
    return transcription.text