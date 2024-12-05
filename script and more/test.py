import numpy as np
import time

import tensorflow as tf
from tensorflow.keras import layers, models
# Load your dataset or generate synthetic data
# For this example, let's assume you have your dataset loaded into X_train and y_train
PI = 3.14159265358979323846
# Evaluate the model on test data

def build_model(config):
    model = models.Sequential()
    model.add(layers.Dense(config[1], activation='relu', input_shape=(1,)))
    for units in config[2:-1]:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1))  # Output layer
    return model

start_time = time.time()
configurations = [[1,200,1],[1, 100,1],[1,100,100,1],[1,50,50,1],[1,25,25,1],[1,50,50,50,50,1],[1,25,25,25,25,25,25,25,25,1]]
file = open('DATA_tensor.txt','w')
file.write('\n')
for conf in configurations:
    file.write(str(conf)+";")
    for trial in range(1):
        x=np.random.uniform(-1*PI,PI, 100)
        trainingsdata = np.sin(x)+np.cos(x*5)
        x2=np.random.uniform(-1*PI,PI, 100)
        valii = np.sin(x2)+np.cos(x2*5)
        netwerk =  build_model(conf)
        netwerk.compile(optimizer='adam',              
              loss='mean_squared_error',  
              metrics=['mae'])
        netwerk.fit(x, trainingsdata, epochs=10, batch_size=32, validation_split=0.2)  # Adjust epochs and batch size as needed
        test_loss, test_acc = netwerk.evaluate(x2, valii)

        file.write(str(test_acc)+";"*int(trial!=199))
        print("trial "+str(trial) +" with "+str(conf))
    file.write("\n")
file.close()
print("--- %s seconds ----- end at: "% (time.time() - start_time), end="")
print(time.asctime)
