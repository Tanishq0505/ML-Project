from flask import Flask,request,render_template# importing flask and other libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__) # creating an instance of Flask class
# __name__ is a special variable in Python that is set to the name of the module in which it is used.

app=application # it is used to create an instance of the Flask class. The instance is assigned to the variable
# app, which is then used to define routes and handle requests.

## Route for a home page

@app.route('/') # it is used to define a route for the home page of the web application.
def index(): #
     # it is used to render the index.html template when the user accesses the home page.
    return render_template('index.html')  # 

@app.route('/predictdata',methods=['GET','POST']) # it is used to define a route for the '/predictdata' URL. The methods=['GET','POST'] argument specifies that this route can handle both GET and POST requests.
def predict_datapoint(): # it is a function that handles the prediction of data points.
    if request.method=='GET': # it checks if the request method is GET. If it is, it renders the home.html template.
        return render_template('home.html') # it is used to render the home.html template when the user accesses the '/predictdata' URL with a GET request.
    else:
        data=CustomData( # it is used to create an instance of the CustomData class. The instance is created with the user input data.
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()  # it is used to convert the user input data into a pandas DataFrame using the get_data_as_data_frame() method of the CustomData class.
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline() # it is used to create an instance of the PredictPipeline class. The instance is used to make predictions on the user input data.
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df) # it is used to make predictions on the user input data using the predict() method of the PredictPipeline class.
        print("after Prediction")
        return render_template('home.html',results=results[0]) # it is used to render the home.html template with the prediction results. The results are passed to the template as a variable named 'results'.
    
if __name__ == "__main__":
    print("Starting Flask App...")
    app.run(host="0.0.0.0") # it is used to run the Flask application. The host is set to "
