from pytorch_grad_cam import GradCAM
from keras.applications import VGG16
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import keras

path='../Face_recognition.h5'
model=keras.models.load_model(path)
target_layer=model.features[-1]
