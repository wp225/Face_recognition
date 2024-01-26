import os
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Flatten, Dropout, Dense, BatchNormalization
from keras_vggface import VGGFace

img_row = 224
img_col = 224

class CustomModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.num_classes = len(os.listdir(self.data_path))

    def base_model(self):
        input_shape = (img_row, img_col, 3)
        base_model = VGGFace(include_top=False,input_shape=input_shape,pooling='max',model='vgg16')

        for layers in base_model.layers:
            layers.trainable = False

        base_model.summary()
        return base_model

    def top_model(self):
        base_model = self.base_model()
        top_model = base_model.output
        top_model = Flatten()(top_model)
        top_model = Dense(256, activation='relu')(top_model)
        top_model = BatchNormalization()(top_model)
        top_model = Dropout(0.5)(top_model)
        top_model = Dense(self.num_classes, activation='softmax')(top_model)

        New_model = Model(inputs=base_model.inputs, outputs=top_model)
        New_model.summary()

        return New_model


if __name__ == '__main__':
    data_path = '/Users/anshujoshi/PycharmProjects/Face_recognition/Test Images'
    Test_model = CustomModel(data_path)
    top_mode = Test_model.top_model()
