import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
class FoodRecommender:
    def __init__(self, food_data_path, model_path, encoder_path):
        # Read the Excel file and pickle files
        self.food_data = pd.read_excel(food_data_path)
        self.model_path = model_path
        self.encoder_path = encoder_path

        with open(self.model_path, 'rb') as model_file:
            self.knn_model = pickle.load(model_file)

        with open(self.encoder_path, 'rb') as encoder_file:
            self.encoder = pickle.load(encoder_file)

    def recommend_similar_foods(self, dish_name, n_recommendations=5):
        dish_name = dish_name.lower()

        if dish_name not in self.food_data['Name'].str.lower().values:
            return f"Dish '{dish_name}' not found in the dataset."

        dish_index = self.food_data[self.food_data['Name'].str.lower() == dish_name].index[0]

        dish_features = self.encoder.transform(
            self.food_data.iloc[[dish_index]][['Ingredients', 'Flavour Profile', 'Course', 'Region', 'State']]
        )

        # Find the nearest neighbors using the KNN model
        distances, indices = self.knn_model.kneighbors(dish_features, n_neighbors=n_recommendations)

        recommended_dishes = self.food_data.iloc[indices[0]]['Name'].values

        return recommended_dishes[:]
  
