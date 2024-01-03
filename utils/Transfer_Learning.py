import os
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Flatten, Dropout, Dense, BatchNormalization

img_row = 224
img_col = 224

train_path = '../Dataset/train'
valid_path = '../Dataset/validation'

class CustomModel:
    def __init__(self, train_path, valid_path):
        self.train_path = train_path
        self.valid_path = valid_path
        self.num_classes = len(os.listdir(self.train_path))

    def base_model(self):
        input_shape=(img_row,img_col,3)
        base_model = VGG16(weights='imagenet', include_top=False,input_shape=input_shape)

        for layer in base_model.layers:
            layer.trainable = False

        base_model.summary()
        return base_model

    def top_model(self):
        base_model = self.base_model()
        top_model = base_model.output
        top_model = Flatten()(top_model)
        top_model = Dense(512, activation='relu')(top_model)
        top_model = BatchNormalization()(top_model)
        top_model = Dropout(0.5)(top_model)
        top_model = Dense(256, activation='relu')(top_model)
        top_model = BatchNormalization()(top_model)
        top_model = Dropout(0.5)(top_model)
        top_model = Dense(self.num_classes, activation='softmax')(top_model)

        New_model=Model(inputs=base_model.inputs,outputs=top_model)
        New_model.summary()

        return New_model

if __name__ == '__main__':
    Test_model = CustomModel(train_path, valid_path)
    top_mode = Test_model.top_model()
