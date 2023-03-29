"# voice-enabled-kiosk" 
I created a mini voice enabled vending kiosk application which will process the user voice input, execute appropriate actions and respond back with voice output. Attached the zip file of demo video clips for various input voice commands.

Brief gist of application:
•	The Streamlit application will detect whether the input voice command is spam or not by an NLP spam classifier. 
•	If input is a valid one, then another classifier namely, action classifier will extract the actions like add, remove, give, display etc. from the input command. 
•	Once the input was validated and action was extracted, the application will identify the metadata like the item name, quantity, units and prepares a document which will be inserted into MongoDB.
•	Eventually, the database operations will be executed based on the input commands

Note: Create a MongoDB instance to store the data
