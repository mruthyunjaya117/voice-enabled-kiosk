from fastapi import FastAPI, Form
from fastapi.responses import FileResponse

import uvicorn

import os
from gtts import gTTS


app = FastAPI()

@app.post('/input/', summary='Please enter the input command')

def display(message :str = Form()):
    
    input_message = message
    
    language = 'en'

    myobj = gTTS(text=input_message, lang=language, slow=False)

    # Saving the converted audio in a mp3 file
    myobj.save("add6.mp3")

# Playing the converted file
    #os.system("start add6.wav")
    return FileResponse("add6.mp3")
    
if __name__ == '__main__':
    uvicorn.run('text_speech:app', host="127.0.0.1", port=6090, reload=True)