#Script
import joblib
import pickle 
import os
print(os.listdir())  # Lists all files in the current directory
# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

user_input=input("enter the mail content")
user_input=[user_input]
#load prefitted vectorizer
feature_extraction = joblib.load("tfidf_vectorizer.pkl")
#predict the results 
user_input=feature_extraction.transform(user_input)

result=model.predict(user_input)
#show the results 
if result[0]==0:
  print("The mail is spam")
else:
  print("the mail is not spam")
