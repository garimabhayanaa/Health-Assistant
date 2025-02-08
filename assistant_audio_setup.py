
import os
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY= os.environ.get("ELEVENLABS_API_KEY")

# Use model for text output to voice
import subprocess
import platform

# Setup text to speech model with gtts
def text_to_speech_with_gtts(input_text,output_filepath):
    language="en"
    audio_obj=gTTS(
        text=input_text,
        lang=language,
        slow= False
    )
    audio_obj.save(output_filepath)
    os_name= platform.system()
    try:
        if os_name=="Darwin":
            subprocess.run(['afplay',output_filepath])
        elif os_name=="Windows":
            subprocess.run(['powershell','-c',f'(New-Object Media.SoundPlayer "{output_filepath}").playSync();'])
        elif os_name=="Linux":
            subprocess.run(['aplay', output_filepath])
        else :
            raise OSError("Unsupported Operating System")
    except Exception as e:
        print(f"An error occured while playing the audio: {e}")

# Setup text to speech model with elevenlabs
def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client= ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio= client.generate(
        text= input_text,
        voice= "Aria",
        output_format= "mp3_22050_32",
        model= "eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath )
    os_name= platform.system()
    try:
        if os_name=="Darwin":
            subprocess.run(['afplay',output_filepath])
        elif os_name=="Windows":
            subprocess.run(['powershell','-c',f'(New-Object Media.SoundPlayer "{output_filepath}").playSync();'])
        elif os_name=="Linux":
            subprocess.run(['aplay', output_filepath])
        else :
            raise OSError("Unsupported Operating System")
    except Exception as e:
        print(f"An error occured while playing the audio: {e}")
    