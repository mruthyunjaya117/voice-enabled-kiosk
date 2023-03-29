'''from fastapi import FastAPI, Form
import speech_recognition as sr
import pyttsx3

import uvicorn

app = FastAPI()

r = sr.Recognizer()

#engine = pyttsx3.init()
#engine.say()
#engine.runAndWait()


try:
    with sr.Microphone() as source2:
        r.adjust_for_ambient_noise(source2, duration=0.2)
        audio2 = r.listen(source2)
        speech = r.recognize_google(audio2)
        speech = speech.lower()
        
        print("The speech ",speech)
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))
#except sr.UnknownValueError:
#    print("unknown error occurred") '''

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import speech_recognition as sr
import uvicorn

app = FastAPI()

class SpeechToText(BaseModel):
    audio: UploadFile

@app.post("/speech-to-text")
def speech_to_text(audio: UploadFile):
    filename = audio.file
    print('audio', audio)
    print('filename', filename)
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio1 = recognizer.record(source)
    text = recognizer.recognize_google(audio1)
    return {"text": text}

if __name__ == "__main__":
    uvicorn.run("speech_text:app", host="127.0.0.1", port=6070, reload=True)
