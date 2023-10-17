"""
predict.py

DESCRIPCIÓN: Script de prediccion para modelo entrenado en el BigMart dataset
AUTOR: Carlos Montiel
FECHA: 17/10/2023
"""

# Imports
import os
import pickle
import pandas as pd

class MakePredictionPipeline(object):
    def __init__(self, input_path, output_path, model_path):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
     
    def load_data(self) -> pd.DataFrame:
        """
        :return data: Loaded data from the feature_engineering.py script 
                      to predict
        :rtype: pd.DataFrame
        """
        data = pd.read_csv(self.input_path+'predict_final.csv')

        return data

    def load_model(self) -> None:
        """
        :return model.pkl: loaded trained model for us as self.model
        :rtype: none
        """ 
        with open(self.model_path+"model.pkl", "rb") as f:
            self.model = pickle.load(f)
        
        return None


    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :return data: Data for prediction with the pred_Sales column 
                      with the prediction
        :rtype: pd.DataFrame
        """ 
        data['pred_Sales'] = self.model.predict(data)

        return data


    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        :return result.csv: File with the data for the prediction and 
                            the predicted data as a new column pred_Sales
        :rtype: none
        """
        with open(self.output_path+"/model/result.csv", 'wb') as file:
                predicted_data.to_csv(file)


    def run(self):
        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_directory = parent_directory.replace("\\", "/")
    pipeline = MakePredictionPipeline(input_path = parent_directory+'/model/',
                                      output_path = parent_directory,
                                      model_path = parent_directory+'/model/')
    pipeline.run()  