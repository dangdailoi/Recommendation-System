import json
import numpy as np
import faiss
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

class ImageSearch:
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)

        self.model_weights_path = config['vgg16_weights']
        self.vectors_file_path = config['images_vector']

        self.model = self._load_feature_extractor()
        self.image_ids, self.image_vectors = self._load_image_features()

    def _load_feature_extractor(self):
        base_model = VGG16(weights=self.model_weights_path)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        return model

    def _load_image_features(self):
        data = np.load(self.vectors_file_path, allow_pickle=True).item()
        image_ids = data['index']
        image_vectors = np.array(data['vector'])
        return image_ids, image_vectors

    def extract_image_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feature_vector = self.model.predict(img_array)
        return feature_vector.flatten()

    def upload_and_search(self, img_path, top_k=48):
        vector_dim = self.image_vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dim)
        index.add(self.image_vectors)

        search_vector = self.extract_image_features(img_path)
        search_vector = np.expand_dims(search_vector, axis=0)

        distances, indices = index.search(search_vector, k=top_k)
        similar_image_ids = [self.image_ids[i] for i in indices[0]]
        return similar_image_ids