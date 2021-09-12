from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

# This is the root page
@app.route('/')
def welcome():
    return "Welcome to new System"

@app.route('/predict')
def predict_note_authentication():  # There are four features
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy  = request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])

    # Returning the prediction

    return "The predicted value is" + str(prediction)

@app.route('/predict',methods=["POST"])
def predict_note_file():  # There are four features
    df_test = pd.read_csv(request.files.get("file"))
    # print(df_test.head())
    prediction = classifier.predict(df_test)

    return str(list(prediction))


if __name__=='__main__':
    app.run()