# Import libraries
import numpy as np
import pandas as pd
from ImageFeatureExtractor import FeatureExtractor
from PIL import Image
import requests
from io import BytesIO

class GenerateSimilarImages:

    # Constructor
    def __init__(self, query_image):
        self.query_image = query_image
        self.feature_extractor = FeatureExtractor(arch='ResNet')

    # Method to generate predictions
    def generate_similar_images(self):

        # load the numpy weights
        saved_features = np.load("model/resnet/resnet_trained_features.npy")
        saved_index = np.load("model/resnet/resnet_trained_index.npy")

        # Load the data
        listing_data = pd.read_csv('FashionImageDataset_FarFetch/listing_data_with_path.csv')

        # '''
        # Read the image URL
        # '''
        # response = requests.get(self.image_url)
        # queryImage = Image.open(BytesIO(response.content))

        # Extracting the Features from the queryImage
        queryImage_features = self.feature_extractor.extract_features(self.query_image)

        # Compute the distance (similarity) bw the query image and the images in the dataset
        # similarity_index = {}
        # for i, feat in zip(saved_index, saved_features):
        #
        #     # Compute the euclidean distance bw the 2 extracted features
        #     distance = np.sum((queryImage_features - feat) ** 2) ** 0.5
        #     similarity_index[i[0]] = distance
        similarity_index = {}
        for i, feat in zip(saved_index, saved_features):
            # Compute the euclidean distance between the two extracted features
            distance = np.sum((queryImage_features - feat) ** 2) ** 0.5
            similarity_index[i] = distance

        # Sort the distances in ascending order and extract the top 10 indexes
        similarity_index_sorted = sorted(similarity_index.items(), key = lambda x : x[1])
        top_8_indexes = [idx for idx, _ in similarity_index_sorted][ : 8]

        # Extract the top 10 images
        top_8_images_path = listing_data.iloc[top_8_indexes]['images.model'].values
        top_8_images_desc = listing_data.iloc[top_8_indexes]['shortDescription'].values

        return self.query_image, top_8_images_path, top_8_images_desc



