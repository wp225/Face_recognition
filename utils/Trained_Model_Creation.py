import os
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from utils.Transfer_Learning import CustomModel

data_path = '/Users/anshujoshi/PycharmProjects/Face_recognition/Dataset/'
class_name = ['Abindra', 'Asmi','Bijen','Jeorge']

class TrainModel:
    def __init__(self, custom_model, data_path='/Users/anshujoshi/PycharmProjects/Face_recognition/Dataset/', epochs=2):
        self.custom_model = custom_model
        self.data_path = data_path
        self.train_batchsize = 16
        self.val_batchsize = 8
        self.epochs = epochs

    def data_loader(self, subset):
        AutoTune = tf.data.AUTOTUNE

        # Apply data augmentation to the training data
        if subset == 'training':
            data_gen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2,  # Moved outside the parentheses
            )
        else:
            data_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)  # Moved outside the parentheses

        data = data_gen.flow_from_directory(
            self.data_path,
            target_size=(224, 224),
            batch_size=self.train_batchsize if subset == 'training' else self.val_batchsize,
            class_mode='categorical',
            shuffle=True,
            seed=123,
            subset=subset,
            classes=class_name
        )

        return data

    def train(self):
        train_generator = self.data_loader(subset='training')
        valid_generator = self.data_loader(subset='validation')

        checkpoint = ModelCheckpoint("/Users/anshujoshi/PycharmProjects/Face_recognition/Face_recognition.h5",
                                     monitor="val_loss",
                                     mode="min",
                                     save_best_only=True,
                                     verbose=1)

        earlystop = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=3,
                                  verbose=1,
                                  restore_best_weights=True)

        tensorboard = TensorBoard(log_dir='../logs', histogram_freq=1, write_graph=True, write_images=True)

        callbacks = [earlystop, checkpoint, tensorboard]

        self.custom_model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

        start_time = datetime.now()
        history = self.custom_model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=valid_generator,
            callbacks=callbacks
        )
        end_time = datetime.now()
        print('Training Time: {}'.format(end_time - start_time))
        return history

if __name__ == '__main__':
    model=CustomModel(data_path)
    model=model.top_model()
    test = TrainModel(model)
    test.train()
