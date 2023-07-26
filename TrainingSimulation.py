print('Setting UP')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from sklearn.model_selection import train_test_split
from keras.models import load_model

#### Step 1: Initialize Data
path = 'Data'
print("Importation Data ... \n")
data = importDataInfo(path)
print("Total Images :", data['Center'].size)
print("End Importation Data ...\n")

#### Step 2: Visualisation and Balancing Data and remove the Redunant Data
print("Start Balancing Data ...")
data = balanceData(data, display=True)
print("End Blacing Data .")

#### Step 3: Prepare for processing
print("Start Preparing Data ...")
imagesPath, steerings = loadData(path, data)
print("End Preparing Data .")
print(imagesPath[0], steerings[0])
print(imagesPath.size)

#### Step 4: Split for Training and Validation
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=10)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))


### Step 8:Creating the Model
print("Start Creating Model ...")
model = creatModel()
print("Creating Model Done")
model.summary()
### Step 9: Training
print("Start Training ...")
history = model.fit(batchGen(xTrain, yTrain, 100, 1),steps_per_epoch=1000,epochs=20,validation_data=batchGen(xVal, yVal, 100, 0),validation_steps=200)
print("Training Done.")
### Step 10: Saving and Plotting

model.save('model.h5')
model = load_model('model.h5')
print('Model Saved')
# plt.plot(model.history['loss'])
# plt.plot(model.history['val_loss'])
# plt.legend(['Training', 'Validation'])
# plt.ylim([0, 1])
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.show()
