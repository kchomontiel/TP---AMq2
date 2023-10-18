"""
train.py

DESCRIPCIÓN: Script de entrenamiento sobre de datos pre-procesados y 
             guardado del mismo
AUTOR: Carlos Montiel, Dario Navarro
FECHA: 17/10/2023
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class ModelTrainingPipeline(object):
    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        :return pandas_df: Dataframe with the pre-processed data for training
        :rtype: pd.DataFrame
        """
        pandas_df = pd.read_csv(self.input_path+'train_final.csv')
        
        return pandas_df

    
    def model_training(self, df: pd.DataFrame):
        """
        :return model_trained: Model trained to use in predictions
        :rtype: pd.DataFrame
        """
        seed = 28
        model = LinearRegression()

        # División de dataset de entrenaimento y validación
        X = df.drop(columns='Item_Outlet_Sales') #[['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
        x_train, x_val, y_train, y_val = train_test_split(X, df['Item_Outlet_Sales'], test_size = 0.3, random_state=seed)

        # Entrenamiento del modelo
        model.fit(x_train,y_train)

        model_trained=model
        
        return model_trained

    def model_dump(self, model_trained) -> None:
        """
        :return model.pkl: Model trained to use in predictions dumped in a file
        :rtype: none
        """
        with open(self.model_path+'model.pkl', 'wb') as file:
            pickle.dump(model_trained, file)
        return None

    def run(self):  
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)

if __name__ == "__main__":
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_directory = parent_directory.replace("\\", "/")
    ModelTrainingPipeline(input_path = parent_directory+'/model/',
                          model_path = parent_directory+'/model/').run()