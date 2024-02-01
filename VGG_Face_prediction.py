import tensorflow as tf
from keras.preprocessing import image
import numpy as np

class Inference:
    def __init__(self, model_path='Face_recognition.h5', class_labels_path='class_labels.txt', image_path='/path/to/inference_image.jpg'):
        self.model_path = model_path
        self.class_labels_path = class_labels_path
        self.image_path = image_path

    def load_model(self):
        model = tf.keras.models.load_model(self.model_path)
        return model

    def load_class_labels(self):
        with open(self.class_labels_path, 'r') as f:
            class_labels = f.read().splitlines()
        return class_labels

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0  # Normalize to [0, 1]

    def infer(self):
        model = self.load_model()
        class_labels = self.load_class_labels()
        preprocessed_img = self.preprocess_image(self.image_path)
        predictions = model.predict(preprocessed_img)

        # Assuming model has been trained for multi-class classification
        predicted_class_index = np.argmax(predictions, axis=1)
        predicted_class_name = class_labels[predicted_class_index[0]]

        print(f"Predicted Class Index: {predicted_class_index[0]}, Predicted Class Name: {predicted_class_name}")
        print(f"Probabilities: {predictions[0]}")

if __name__ == '__main__':
    inference_image_path = '/Users/anshujoshi/PycharmProjects/Face_recognition/Dataset/1.jpg'  # Replace with the path to your inference image
    inference = Inference(model_path='/Users/anshujoshi/PycharmProjects/Face_recognition/utils/Face_recognition.h5', class_labels_path='/Users/anshujoshi/PycharmProjects/Face_recognition/utils/class_labels.txt', image_path=inference_image_path)
    inference.infer()
