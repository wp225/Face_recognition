import os
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from utils.Transfer_Learning import CustomModel


class TrainModel:
    def __init__(self, custom_model, data_path,epochs=10):
        self.custom_model = custom_model
        self.data_path = data_path
        self.train_batchsize = 16
        self.val_batchsize = 8
        self.epochs = epochs

    def data_loader(self, subset):
        AutoTune = tf.data.AUTOTUNE

        data_gen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2,
        ) if subset == 'training' else ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

        data = data_gen.flow_from_directory(
            self.data_path,
            target_size=(224, 224),
            batch_size=self.train_batchsize if subset == 'training' else self.val_batchsize,
            class_mode='categorical',
            shuffle=True,
            seed=123,
            subset=subset
        )

        return data

    def train(self):
        train_generator = self.data_loader(subset='training')
        valid_generator = self.data_loader(subset='validation')

        # Get the class labels from the generator
        class_labels = list(train_generator.class_indices.keys())

        checkpoint = ModelCheckpoint("Face_recognition.h5",
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

        self.custom_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
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

        # Save the class labels to a file for later use during inference
        with open('class_labels.txt', 'w') as f:
            f.write("\n".join(class_labels))

        return history


if __name__ == '__main__':
    data_path = '../Test Images'
    model = CustomModel(data_path)
    model = model.top_model()
    trainer = TrainModel(model)
    trainer.train()
