from utils.Transfer_Learning import CustomModel
from utils.Trained_Model_Creation import TrainModel


data='/Users/anshujoshi/PycharmProjects/Face_recognition/Dataset'

model=CustomModel(data)
full_model=model.top_model()

training=TrainModel(full_model,data,epochs=10)
ret=training.train()
