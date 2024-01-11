import cv2
import numpy as np
import tensorflow as tf
import keras
import os
class_name = ['Abindra', 'Asmi','Bijen','Jeorge']
model=keras.models.load_model('/Users/anshujoshi/PycharmProjects/Face_recognition/Face_recognition.h5')
# Load the pre-trained model
for files in os.listdir('/Users/anshujoshi/PycharmProjects/Face_recognition/Dataset/Jeorge'):
    image_path=os.path.join('/Users/anshujoshi/PycharmProjects/Face_recognition/Dataset/Jeorge',files)
    image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    image_array /= 255.0  # Normalize pixel values to be between 0 and 1

    predictions = model.predict(image_array)
    print(predictions)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_name[predicted_class_index]

    print(predicted_class_name)
