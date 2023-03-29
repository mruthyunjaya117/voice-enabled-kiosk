
from gtts import gTTS

import os

# The text that you want to convert to audio
mytext = 'display the current items in inventory'


language = 'en'

myobj = gTTS(text=mytext, lang=language, slow=False)

# Saving the converted audio in a mp3 file
myobj.save("add5.mp3")

# Playing the converted file
#os.system("start add.wav")

from pydub import AudioSegment
import pydub


filename = "add5.mp3"  # File that already exists.
#filename = './notinmenu.wav'
#print(os.getcwd())
                       
sound = AudioSegment.from_mp3(filename)
sound.export('formatted_display.wav', format="wav")
print("The audio was generated successfully")