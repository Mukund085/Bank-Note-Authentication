from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger

app=Flask(__name__)
Swagger(app) #This will generate the UI using a different URL

pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

# This is the root page
@app.route('/')
def welcome():
    return "Welcome to new System"

@app.route('/predict')
def predict_note_authentication():  # There are four features

    """Let's Authenticate the Banks Note
        This is using docstrings for specifications.
        ---
        parameters:
          - name: variance
            in: query
            type: number
            required: true
          - name: skewness
            in: query
            type: number
            required: true
          - name: curtosis
            in: query
            type: number
            required: true
          - name: entropy
            in: query
            type: number
            required: true
        responses:
            200:
                description: The output values

        """


    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy  = request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "The value is" + str(prediction)

@app.route('/predict',methods=["POST"])
def predict_note_file():  # There are four features

    """Let's Authenticate the Banks Note
            This is using docstrings for specifications.
            ---
            parameters:
              - name: file
                in: formData
                type: file
                required: true

            responses:
                200:
                    description: The output values

    """



    df_test = pd.read_csv(request.files.get("file"))
    # print(df_test.head())
    prediction = classifier.predict(df_test)

    return str(list(prediction))


if __name__=='__main__':
    app.run()