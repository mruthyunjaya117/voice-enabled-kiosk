import streamlit as st
import speech_recognition as sr

r = sr.Recognizer()

st.title("Speech to Text")

if st.button("Start Recording"):
    with st.spinner("## Please speak"):
        with sr.Microphone() as source:
            audio = r.listen(source)
            a =1
        try:
            text = r.recognize_google(audio)
            st.success("Recognized: " + text)
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError as e:
            st.error("Error: {0}".format(e))

    if a==1:
        st.write(" the text is", text)