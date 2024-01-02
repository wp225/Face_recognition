import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from Transfer_Learning import CustomModel  # Assuming CustomModel is defined in Transfer_Learning module

img_row = 224
img_col = 224
train_path = '../Dataset/train'
valid_path = '../Dataset/validation'


class TrainModel:

    def __init__(self, custom_model, train_path, valid_path, epochs):
        self.custom_model = custom_model
        self.train_path = train_path
        self.valid_path = valid_path
        self.train_batchsize = 16
        self.val_batchsize = 8
        self.epochs = epochs

    def data_loader(self):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 225,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 225)

        train_generator = train_datagen.flow_from_directory(self.train_path,
                                                            target_size=(img_row, img_col),
                                                            batch_size=self.train_batchsize,
                                                            class_mode='categorical',
                                                            shuffle=True)

        valid_generator = valid_datagen.flow_from_directory(self.valid_path,
                                                            target_size=(img_row, img_col),
                                                            batch_size=self.val_batchsize,
                                                            class_mode='categorical',
                                                            shuffle=False
                                                            )

        return train_generator, valid_generator

    def train(self):
        checkpoint = ModelCheckpoint("../face_recog.h5",
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

        # Ensure that the custom_model has a compile method
        self.custom_model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(lr=0.001),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

        train_generator, valid_generator = self.data_loader()

        history = self.custom_model.fit(
            train_generator,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=valid_generator,
        )

        return history


if __name__ == '__main__':
    custom_model_instance = CustomModel(train_path, valid_path)
    full_model = custom_model_instance.top_model()

    training_instance = TrainModel(custom_model_instance, train_path, valid_path, epochs=10)
    training_result = training_instance.train()
