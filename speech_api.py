from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse

from gtts import gTTS

from pydantic import BaseModel
import speech_recognition as sr

import uvicorn

import os

import pandas as pd
pd.set_option('display.max_colwidth', 200)

import numpy as np
import re

# for NLP related tasks
import spacy
global nlp
nlp=spacy.load('en_core_web_sm')

# for mongodb operations
from pymongo import MongoClient

# saving model as pickle
import pickle

import json as js

description = """
This endpoint will accepts user input commands and returns inventory data as JSON response. 🚀 \n
**contact**: Dhanunjaya Raga <dhanunjaya.raga@in.bosch.com>
"""
def text_cleaner(text):
  
  #remove user mentions
    text = re.sub(r'@[A-Za-z0-9]+','',text)           
  
  #remove hashtags
  #text = re.sub(r'#[A-Za-z0-9]+','',text)         
  
  #remove links
    text = re.sub(r'http\S+', '', text)  

  #convering text to lower case
    text = text.lower()

  # fetch only words
    text = re.sub("[^a-z]+", " ", text)

  # removing extra spaces
    text=re.sub("[\s]+"," ",text)
  
  # creating doc object
    doc=nlp(text)

  # remove stopwords and lemmatize the text
    tokens=[token.lemma_ for token in doc if(token.is_stop==False)]
  
  #join tokens by space
    return " ".join(tokens)
    
df = pd.read_csv(r'C:/Users/DAG9KOR/Downloads/ProjectMulticlasstextclassification/inventory.csv')

# perform text cleaning
df['clean_text']= df['text'].apply(text_cleaner)

# getting the values of required columns
text   = df['clean_text'].values
labels = df['label'].values
actions = df['action'].values

#importing label encoder
from sklearn.preprocessing import LabelEncoder

#define label encoder
le = LabelEncoder()
le1 = LabelEncoder()

#fit and transform target strings to a numbers
labels = le.fit_transform(labels)
actions = le1.fit_transform(actions)

# instantiating the word_vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

pickled_model = pickle.load(open('nb_model.pkl', 'rb'))
pickled_vectorizer = pickle.load(open('vectorizer.pkl','rb'))

pickled_model_action = pickle.load(open('nb_model_action.pkl', 'rb'))
pickled_vectorizer_action = pickle.load(open('vectorizer_action.pkl','rb'))

class SpeechToText(BaseModel):
    audio: UploadFile

# Instantiating the FastAPI object
app = FastAPI(title='Inventory Kiosk',description=description)

def text_to_speech(data):

    input_message = data
    
    language = 'en'

    myobj = gTTS(text=input_message, lang=language, slow=False)

    # Saving the converted audio in a mp3 file
    myobj.save("speech_response.mp3")

    return FileResponse("speech_response.mp3")
    

@app.get("/health" , summary="Returns a simple and valid response for the environment to check if the application is still healthy")

def welcome():
    return { "response":"The service endpoint is up and running"}
    
@app.post('/input/', summary='Please enter the input command')

def display(audio: UploadFile):
    
    filename = audio.file
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio1 = recognizer.record(source)
    text = recognizer.recognize_google(audio1)
    #input_message = 'give 300 kg of Sandwich from inventory'
    #input_message = 'update 20 kg of Biscuits to stocks inventory'
    print("The text from audio", text)
    input_message = text.lower()
    print("the input message", input_message)
    #input_message = 'what is the gdp of india'
    #input_message = 'please add me to your fb account'
    #input_message = "remove 12 kg of Sugar to food category"
    #input_message = "update inventory by 5 kg of Sugar"
    #input_message = 'what do you offer for me'
    #input_message = 'display the existing data'
    
    # predicting the label from input message
    processed = text_cleaner(input_message)
    vector = pickled_vectorizer.transform([processed])
    pred = pickled_model.predict(vector)
        
    label = le.inverse_transform(np.array(pred))

    # predicting the action from input message
    vector_action = pickled_vectorizer_action.transform([processed])
    pred_action = pickled_model_action.predict(vector_action)
    #print("the pred_action--->", pred_action)
    action_label = le1.inverse_transform(np.array(pred_action))[0]
    print('action label: ', action_label)
    
    
    # available menu
    menu = ['Biscuits','Milk','Sandwich','Fruits','Wheat','Sugar','Salt','Bread','Detergent','Softdrinks','Sweets']

    # actions that can be performed with inventory
    add_action = ['add','append','push']
    remove_action = ['remove','delete','subtract']
    display_action = ['display','provide','show','offer','retrieve','extract','get']
    give_action = ['give','dispatch','dispense']

    json = {}

    try:

        if label == 'ham':
            print(f"The input message '{input_message}' is valid")

            # database connection
            uri = "mongodb://dhanu:dhanu@localhost:27072/?authSource=admin"
            client = MongoClient(uri)
            db = client['inventory']
            collection = db['products']

            # spaCy object creation
            doc = nlp(input_message)

            # identifying the quantity entities using NER
            for ent in doc.ents:
                if ent.label_ == 'QUANTITY':
                    item_quantity = re.search('\d+', ent.text)
                    item_quantity = item_quantity.group()
                    json['item_quantity'] = int(item_quantity)
                    #print("the quantity----->",json['item_quantity'])
                    item_units = re.search('\D+', ent.text)
                    item_units = str(item_units.group())
                    json['units'] = item_units.strip()
                    #print("The units are ----->",json['units'])

                elif ent.label_ == 'CARDINAL':
                    item_quantity = int(ent.text)
                    #print("The cardinal number--->",item_quantity)
                    json['item_quantity'] = item_quantity
                    json['units'] = 'NA'


            # extracting the item from input message
            for token in doc:
                #print(token)
                for i in menu:
                    if token.text.lower() == i.lower():
                        item1 = menu[menu.index(i)]
                        json['item'] = item1


            # identifying the action from input message
            action = []
            for token in doc:
                if token.pos_ == 'VERB':
                    action.append(token.text)

            print("The action from input message derived from POS: ",action[0])


            # display action processing
            if action_label in display_action:
                print("The following items are present in the inventory:\n")
                cursor = collection.find({},{'_id':0})
                item_list = []
                for itr in cursor:
                    item_list.append(itr)

                df_items = pd.DataFrame(item_list)
                #print(df_items)
                temp = df_items.to_json(orient = "records")
                parsed = js.loads(temp)
                #return parsed
                voice = df_items.to_string()
                output = text_to_speech(voice)
                return output
            
            if action_label == 'update':
                #return f"The update action is anonymous. Please add or remove the items from inventory"
                voice = f"The update action is anonymous. Please add or remove the items from inventory"
                output = text_to_speech(voice)
                return output

            # input products check in the inventory
            elif json.get('item') == None:
                print("The specified item from input message is not in the Menu. The available menu: \n", menu)
                #return f"The specified item from input message is not in the Menu. The available menu: {menu}"
                voice = f"The specified item from input message is not in the Menu. The available menu: {menu}"
                output = text_to_speech(voice)
                return output

            else:
                print("The metadata extracted from input message:\n", json)

            # add action process
            if action_label in add_action:

                if json['units'] == 'kg' and json.get('item'):

                    # filter for searching the item
                    search_filter = {'item':json['item'], 'units':'kg'}
                    
                    fetch = collection.find_one(search_filter)
                    if fetch:
                        response = {}
                        print("Available {} stock: {} {}".format(json['item'],fetch['item_quantity'],json['units']))
                        
                        response['existing'] = f"Available {json['item']} stock: {fetch['item_quantity']} {json['units']}"
                        db_quantity = fetch['item_quantity']

                        #quantity extracted from input message
                        quantity = {'$inc':{'item_quantity':json['item_quantity']}}

                        # database operation
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        after = collection.find_one(search_filter)
                        response['after'] = f"Available {json['item']} stock after processing the command: {after['item_quantity']} {json['units']}"
                        response['input'] = input_message
                        
                        voice = f"Available {json['item']} stock: {fetch['item_quantity']} {json['units']}. Available {json['item']} stock after processing the command: {after['item_quantity']} {json['units']}"
                        
                        output = text_to_speech(voice)
                        return output
                        #return response
                    else:
                        quantity = {'$inc':{'item_quantity':json['item_quantity']}}
                        
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        current = collection.find_one(search_filter)
                        
                        #return f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} {json['units']}"
                        
                        voice = f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} {json['units']}"
                        output = text_to_speech(voice)
                        return output
                        
                       

                elif json['units'] == 'liter' and json.get('item'):

                    # filter for searching the item
                    search_filter = {'item':json['item'], 'units':'liter'}
                    
                    fetch = collection.find_one(search_filter)
                    
                    if fetch:
                        response = {}
                        print("Available {} stock: {} {}".format(json['item'],fetch['item_quantity'],json['units']))
                        
                        response['existing'] = f"Available {json['item']} stock: {fetch['item_quantity']} {json['units']}"
                        db_quantity = fetch['item_quantity']

                        #quantity extracted from input message
                        quantity = {'$inc':{'item_quantity':json['item_quantity']}}

                        # database operation
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        after = collection.find_one(search_filter)
                        response['after'] = f"Available {json['item']} stock after processing the command: {after['item_quantity']} {json['units']}"
                        response['input'] = input_message
                        
                        voice = f"Available {json['item']} stock: {fetch['item_quantity']} {json['units']}. Available {json['item']} stock after processing the command: {after['item_quantity']} {json['units']}"
                        
                        output = text_to_speech(voice)
                        return output
                        #return response
                        
                    else:
                        quantity = {'$inc':{'item_quantity':json['item_quantity']}}
                        
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        current = collection.find_one(search_filter)
                        
                        #return f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} {json['units']}"
                        voice = f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} {json['units']}"

                        output = text_to_speech(voice)
                        return output
                  
                elif json['units'] == 'NA' and json.get('item'):

                    search_filter = {'item':json['item'], 'units':'NA'}
                    
                    fetch = collection.find_one(search_filter)
                    
                    if fetch:
                        response = {}
                        print("Available {} stock: {} {}".format(json['item'],fetch['item_quantity'],json['units']))
                        
                        response['existing'] = f"Available {json['item']} stock: {fetch['item_quantity']} units"
                        db_quantity = fetch['item_quantity']

                        #quantity extracted from input message
                        quantity = {'$inc':{'item_quantity':json['item_quantity']}}

                        # database operation
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        after = collection.find_one(search_filter)
                        response['after'] = f"Available {json['item']} stock after processing the command: {after['item_quantity']} units"
                        response['input'] = input_message
                        
                        #return response
                        voice = f"Available {json['item']} stock: {fetch['item_quantity']} units. Available {json['item']} stock after processing the command: {after['item_quantity']} units"

                        output = text_to_speech(voice)
                        return output
                     
                    else:
                        quantity = {'$inc':{'item_quantity':json['item_quantity']}}
                        
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        current = collection.find_one(search_filter)
                        
                        #return f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} units"

                        voice = f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} units"

                        output = text_to_speech(voice)
                        return output


                     
                else:
                    print("The product from input message was not available in inventory")
                    #return f"The product from input message was not available in inventory"
                    voice = f"The product from input message was not available in inventory"
                    output = text_to_speech(voice)
                    return output

            # delete action process
            elif action_label in remove_action:

                if json['units'] == 'kg' and json.get('item'):

                    # filter for searching the item
                    search_filter = {'item':json['item'], 'units':'kg'}

                    fetch = collection.find_one(search_filter)
                     
                    if fetch:
                        response = {}
                        print("Available {} stock: {} {}".format(json['item'],fetch['item_quantity'],json['units']))
                        
                        response['existing'] = f"Available {json['item']} stock: {fetch['item_quantity']} {json['units']}"
                        db_quantity = fetch['item_quantity']

                        #quantity extracted from input message
                        quantity = {'$inc':{'item_quantity':-json['item_quantity']}}

                        # database operation
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        after = collection.find_one(search_filter)
                        response['after'] = f"Available {json['item']} stock after processing the command: {after['item_quantity']} {json['units']}"
                        response['input'] = input_message
                        
                        #return response
                        voice = f"Available {json['item']} stock: {fetch['item_quantity']} {json['units']}. Available {json['item']} stock after processing the command: {after['item_quantity']} {json['units']}"

                        output = text_to_speech(voice)
                        return output
                    else:
                        quantity = {'$inc':{'item_quantity':-json['item_quantity']}}
                        
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        current = collection.find_one(search_filter)
                        
                        #return f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} {json['units']}"

                        voice = f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} {json['units']}"

                        output = text_to_speech(voice)
                        return output

                elif json['units'] == 'liter' and json.get('item'):

                    # filter for searching the item
                    search_filter = {'item':json['item'], 'units':'liter'}
                    
                    if fetch:
                        response = {}
                        print("Available {} stock: {} {}".format(json['item'],fetch['item_quantity'],json['units']))
                        
                        response['existing'] = f"Available {json['item']} stock: {fetch['item_quantity']} {json['units']}"
                        db_quantity = fetch['item_quantity']

                        #quantity extracted from input message
                        quantity = {'$inc':{'item_quantity':-json['item_quantity']}}

                        # database operation
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        after = collection.find_one(search_filter)
                        response['after'] = f"Available {json['item']} stock after processing the command: {after['item_quantity']} {json['units']}"
                        response['input'] = input_message
                        
                        #return response
                        voice = f"Available {json['item']} stock: {fetch['item_quantity']} {json['units']}. Available {json['item']} stock after processing the command: {after['item_quantity']} {json['units']}"

                        output = text_to_speech(voice)
                        return output
                        
                    else:
                        quantity = {'$inc':{'item_quantity':-json['item_quantity']}}
                        
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        current = collection.find_one(search_filter)
                        
                        #return f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} {json['units']}"

                        voice = f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} {json['units']}"

                        output = text_to_speech(voice)
                        return output
                    

                elif json['units'] == 'NA' and json.get('item'):

                    search_filter = {'item':json['item'], 'units':'NA'}

                    fetch = collection.find_one(search_filter)
                    
                    if fetch:
                        response = {}
                        print("Available {} stock: {} {}".format(json['item'],fetch['item_quantity'],json['units']))
                        
                        response['existing'] = f"Available {json['item']} stock: {fetch['item_quantity']} units"
                        db_quantity = fetch['item_quantity']

                        #quantity extracted from input message
                        quantity = {'$inc':{'item_quantity':-json['item_quantity']}}

                        # database operation
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        after = collection.find_one(search_filter)
                        response['after'] = f"Available {json['item']} stock after processing the command: {after['item_quantity']} units"
                        response['input'] = input_message
                        #return response

                        voice = f"Available {json['item']} stock: {fetch['item_quantity']} units.Available {json['item']} stock after processing the command: {after['item_quantity']} units"

                        output = text_to_speech(voice)
                        return output
                     
                    else:
                        quantity = {'$inc':{'item_quantity':-json['item_quantity']}}
                        
                        collection.update_one(search_filter, quantity, upsert=True)
                        
                        current = collection.find_one(search_filter)
                        
                        #return f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} units"

                        voice = f"The items are currently not present in inventory. They are freshly added to the inventory. Available {json['item']} stock: {current['item_quantity']} units"

                        output = text_to_speech(voice)
                        return output

                else:
                    print("The product from input message was not available in inventory")
                    #return f"The product from input message was not available in inventory"
                    voice = f"The product from input message was not available in inventory"
                    output = text_to_speech(voice)
                    return output

            # dispatch action processing
            elif action_label in give_action:

                if json['units'] == 'kg' and json.get('item'):

                    # filter for searching the item
                    search_filter = {'item':json['item'], 'units':'kg'}

                    # fetching the documents from db
                    cursor = collection.find_one(search_filter)
                    if cursor:
                        print("Available {} stock: {} {}".format(json['item'],cursor['item_quantity'],json['units']))
                        db_quantity = cursor['item_quantity']

                        if json['item_quantity'] > db_quantity:
                            print("Insufficient items in inventory")
                            #return f"Insufficient items in inventory. Available {json['item']} stock: {cursor['item_quantity']} {json['units']}"

                            voice = f"Insufficient items in inventory. Available {json['item']} stock: {cursor['item_quantity']} {json['units']}"

                            output = text_to_speech(voice)
                            return output
                            
                        else:
                            print("The items are available and ready to dispense")
                            
                            
                            quantity = {'$inc':{'item_quantity':-json['item_quantity']}}
                            collection.update_one(search_filter, quantity, upsert=True)
                            
                            query = collection.find_one(search_filter)
                            #return f"The items are available and ready to dispense. Available {json['item']} stock: {cursor['item_quantity']} {json['units']}. The quantity after dispensing the {json['item']} is {query['item_quantity']} kg"

                            voice = f"The items are available and ready to dispense. Available {json['item']} stock: {cursor['item_quantity']} {json['units']}. The quantity after dispensing the {json['item']} is {query['item_quantity']} kg"

                            output = text_to_speech(voice)
                            return output
                                 
                    else:
                        print(f"The desired item '{json['item']}' is not available. Please add to inventory")
                        #return f"The desired item '{json['item']}' is not available. Please add to inventory"
                        voice = f"The desired item '{json['item']}' is not available. Please add to inventory"

                        output = text_to_speech(voice)
                        return output

                elif json['units'] == 'liter' and json.get('item'):

                    # filter for searching the item
                    search_filter = {'item':json['item'], 'units':'liter'}

                    # fetching the documents from db
                    cursor = collection.find_one(search_filter)
                    if cursor:
                        print("Available {} stock: {} {}".format(json['item'],cursor['item_quantity'],json['units']))
                        db_quantity = cursor['item_quantity']

                        if json['item_quantity'] > db_quantity:
                            print("Insufficient items in inventory")
                            return f"Insufficient items in inventory. Available {json['item']} stock: {cursor['item_quantity']} {json['units']}"
                        else:
                            print("The items are available and ready to dispense")
                            
                            quantity = {'$inc':{'item_quantity':-json['item_quantity']}}
                            collection.update_one(search_filter, quantity, upsert=True)
                            
                            query = collection.find_one(search_filter)
                            
                            #return f"The items are available and ready to dispense. Available {json['item']} stock: {cursor['item_quantity']} {json['units']}. The quantity after dispensing the {json['item']} is {query['item_quantity']} liters"

                            voice = f"The items are available and ready to dispense. Available {json['item']} stock: {cursor['item_quantity']} {json['units']}. The quantity after dispensing the {json['item']} is {query['item_quantity']} liters"

                            output = text_to_speech(voice)
                            return output
                    else:
                        print(f"The desired item '{json['item']}' is not available. Please add to inventory")
                        #return f"The desired item '{json['item']}' is not available. Please add to inventory"

                        voice = f"The desired item '{json['item']}' is not available. Please add to inventory"
                        output = text_to_speech(voice)
                        return output

                elif json['units'] == 'NA' and json.get('item'):

                     # filter for searching the item
                    search_filter = {'item':json['item'], 'units':'NA'}

                    # fetching the documents from db
                    cursor = collection.find_one(search_filter)

                    if cursor:
                        print("Available {} stock: {} ".format(json['item'],cursor['item_quantity']))
                        db_quantity = cursor['item_quantity']

                        if json['item_quantity'] > db_quantity:
                            print("Insufficient items in inventory")
                            return f"Insufficient items in inventory. Available {json['item']} stock: {cursor['item_quantity']}"
                        else:
                            print("The items are available and ready to dispense")
                            
                            quantity = {'$inc':{'item_quantity':-json['item_quantity']}}
                            collection.update_one(search_filter, quantity, upsert=True)
                            
                            query = collection.find_one(search_filter)
                            #return f"The items are available and ready to dispense. Available {json['item']} stock: {cursor['item_quantity']} {json['units']}. The quantity after dispensing the {json['item']} is {query['item_quantity']}"

                            voice = f"The items are available and ready to dispense. Available {json['item']} stock: {cursor['item_quantity']} {json['units']}. The quantity after dispensing the {json['item']} is {query['item_quantity']}"

                            output = text_to_speech(voice)
                            return output
                    else:
                        print(f"The desired item '{json['item']}' is not available. Please add to inventory")
                        #return f"The desired item '{json['item']}' is not available. Please add to inventory"
                        voice = f"The desired item '{json['item']}' is not available. Please add to inventory"

                        output = text_to_speech(voice)
                        return output

            else:
                print("There is no action from input message")
                #return f"There is no action from input message"

                voice = f"There is no action from input message"
                output = text_to_speech(voice)
                return output


        else:
            print(f"The input message '{input_message}' was not valid")
            #return f"The input message '{input_message}' was not valid"
            voice = f"The input message '{input_message}' was not valid"
            output = text_to_speech(voice)
            return output


    except Exception as error:
         print("The exception is --->", error)
    

if __name__ == "__main__":
    uvicorn.run("speech_api:app", host="127.0.0.1", port=6080, reload=True)