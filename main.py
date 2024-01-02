from utils.Transfer_Learning import CustomModel
from utils.Trained_Model_Creation import TrainModel

train_path = './Dataset/train'
valid_path = './Dataset/validation'

model=CustomModel(train_path,valid_path)
full_model=model.top_model()

training=TrainModel(full_model,train_path,valid_path,epochs=10)
ret=training.train()
