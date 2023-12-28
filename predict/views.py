from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
import pandas as pd
import pickle as pkl

# Create your views here.
class Titanic(APIView):
    def post(self, request):
         # getting features from user
         Pclass= request.data["Pclass"]
         Name = request.data["Name"]
         Sex = request.data["Sex"]
         Age = request.data["Age"]
         SibSp  = request.data["SibSp"]
         Parch = request.data["Parch"]
         Ticket = request.data["Ticket"]
         Fare  = request.data["Fare"]
         Cabin = request.data["Cabin"]
         Embarked  = request.data["Embarked"]
         
         
        # convert features into dataframe
         df = pd.DataFrame ({
            "Pclass": Pclass,
            "Name": Name,
            "Sex": Sex,
            "Age": Age,
            "SibSp": SibSp,
            "Parch":Parch,
            "Ticket": Ticket,
            "Fare":Fare,
            "Cabin": Cabin,
            "Embarked": Embarked

        }, index = [0])
         
         # import one hot encoder and encode the categorical variables
         with open ("./Encoder/OneHotEncoder.pkl", "rb") as f:
               encoders = pkl.load(f)
         df_encode = encoders.transform(df[["Name", "Sex", "Ticket", "Cabin", "Embarked"]])
         feature_names = encoders.get_feature_names_out(["Name", "Sex", "Ticket", "Cabin", "Embarked"])
         df_encode = pd.DataFrame(df_encode, columns = feature_names)
         df = pd.concat([df, df_encode], axis = 1).drop(columns = ["Name", "Sex", "Ticket", "Cabin", "Embarked"])
        
         # import rf classifer and make prediction
         with open ("./Models/RandomForest.pkl", "rb") as f:
               rf_model = pkl.load(f)
           
         pred = rf_model.predict(df)

         return Response ({
           "RandomForest" : encoders.inverse_transform(pred)
        })
 
