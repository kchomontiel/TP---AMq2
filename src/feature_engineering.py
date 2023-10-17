"""
feature_engineering.py

DESCRIPCIÓN: Script de transformacion de datos para el entrenamiento
AUTOR: Carlos Montiel
FECHA: 17/10/2023
"""

# Imports
import os
import pandas as pd
import json
import tempfile
import argparse


class FeatureEngineeringPipeline(object):
    def __init__(self, is_train, input_path, output_path):
        self.is_train = is_train
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
        if(self.is_train):
            data_train = pd.read_csv(self.input_path+'/data/Train_BigMart.csv')
            data_test = pd.read_csv(self.input_path+'/data/Test_BigMart.csv')
            data_train['Set'] = 'train'
            data_test['Set'] = 'test'
            pandas_df = pd.concat([data_train, data_test], ignore_index=True, 
                                  sort=False)
        else:
            print("is_train false: reading json...")
            with open(self.input_path+'/Notebook/example.json', 'r', 
                      encoding="utf-8") as f:
                data = json.load(f)
            pandas_df = pd.DataFrame([data])
        return pandas_df

    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :return df_transformed: Transformed df for training as a DataFrame
        :rtype: pd.DataFrame
        """
        products = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
        for product in products:
            mean = (data[data['Item_Identifier'] == product][['Item_Weight']]).mode().iloc[0,0]
            data.loc[data['Item_Identifier'] == product, 'Item_Weight'] = mean
        outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        for outlet in outlets:
            data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

        dataframe = data.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()
        dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace(
                                        {'High': 2, 'Medium': 1, 'Small': 0})
        dataframe['Outlet_Location_Type'] = dataframe[
                                        'Outlet_Location_Type'].replace(
                                        {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})
        dataframe = pd.get_dummies(dataframe, columns=['Outlet_Type'])
        if(not self.is_train):
            defaults = {
                'Outlet_Type_Supermarket Type1': False,
                'Outlet_Type_Supermarket Type2': False,
                'Outlet_Type_Supermarket Type3': False,
                'Outlet_Type_Grocery Store': False
            }
            template_df = pd.read_csv(self.input_path+'/model/test_final.csv')
            template_df = pd.DataFrame(columns=template_df.columns)
            print("Template: ",template_df.columns)
            for column in dataframe.columns:
                template_df[column] = dataframe[column]
            template_df = template_df.fillna(defaults)
            template_df = template_df.drop(columns='Unnamed: 0', axis=1)
            dataframe = template_df
        df_transformed = dataframe.drop(columns=['Item_Identifier',
                                                  'Outlet_Identifier'])
        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        :return train_final.csv, test_final.csv: Files used for testing and 
                                                model training
        :rtype: none
        """
        if(self.is_train):
            df_train = transformed_dataframe.loc[transformed_dataframe['Set'] == 'train']
            df_test = transformed_dataframe.loc[transformed_dataframe['Set'] == 'test']
            df_train.drop(['Set'], axis=1, inplace=True)
            df_test.drop(['Item_Outlet_Sales','Set'], axis=1, inplace=True)
            with open(self.output_path+"train_final.csv", 'wb') as file:
                df_train.to_csv(file)
            with open(self.output_path+"test_final.csv", 'wb') as file:
                df_test.to_csv(file)
        else:
            print("is_train false: saving predict df")
            with open(self.output_path+"predict_final.csv", 'wb') as file:
                transformed_dataframe.to_csv(file)


    def run(self):
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descripción de tu script")
    parser.add_argument("is_train", type=str, help="Boolean to set variables"
                        +"for training or for prediction")
    args = parser.parse_args()
    print("Args value:",args.is_train)
    if args.is_train == "True":
        args.is_train = True
    elif args.is_train == "False":
        args.is_train = False

    parent_directory = os.path.dirname(os.path.dirname(
                                                    os.path.abspath(__file__)))
    parent_directory = parent_directory.replace("\\", "/")
    FeatureEngineeringPipeline( is_train = args.is_train,
                                input_path = parent_directory,
                                output_path = parent_directory+'/model/').run()