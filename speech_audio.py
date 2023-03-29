
import speech_recognition as sr
import pyttsx3
import os

r = sr.Recognizer()

#engine = pyttsx3.init()
#engine.say()
#engine.runAndWait()


try:
    with sr.Microphone() as source2:
        r.adjust_for_ambient_noise(source2, duration=0.2)
        audio2 = r.listen(source2)
        audio2.save(add.mp3)
        speech = r.recognize_google(audio2)
        speech = speech.lower()
        
        
        print("The speech ",speech)
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))
except sr.UnknownValueError:
    print("unknown error occurred")