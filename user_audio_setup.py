import logging
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
from io import BytesIO
from pydub import AudioSegment
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def record_audio(file_path="recorded_audio.mp3", duration=10, sample_rate=44100):
    """
    Records audio using sounddevice and saves it as an MP3 file.
    
    Args:
        file_path (str): Path to save the recorded audio.
        duration (int): Recording duration in seconds.
        sample_rate (int): Sample rate for recording.
    
    Returns:
        str: File path of the recorded audio.
    """
    try:
        logging.info("Recording... Speak now!")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()
        wav_buffer = BytesIO()
        wav.write(wav_buffer, sample_rate, audio_data)
        wav_buffer.seek(0)
        
        # Convert recorded audio to MP3
        audio_segment = AudioSegment.from_wav(wav_buffer)
        audio_segment.export(file_path, format='mp3', bitrate='128K')
        logging.info(f"Audio saved to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"An error occurred while recording: {e}")
        return None

def transcribe_with_groq(audio_filepath, GROQ_API_KEY):
    """
    Transcribes recorded audio using Groq's Whisper model.
    
    Args:
        audio_filepath (str): Path to the recorded audio file.
        GROQ_API_KEY (str): API key for Groq.
    
    Returns:
        str: Transcribed text.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        stt_model = "whisper-large-v3"
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        return transcription.text
    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return None

# Example usage
audio_file = record_audio(duration=10)
if audio_file:
    transcription = transcribe_with_groq(audio_file, GROQ_API_KEY)
    if transcription:
        print("Transcription:", transcription)
