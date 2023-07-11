from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

# Loading the Data
Data=pd.read_csv("Data.csv")
Data = Data[['u_propo', 'u_remi', 'BIS', 'x_propo_1', 'x_propo_4', 'x_remi_1', 'x_remi_4']]

# Creating the Training/Test Sets
x = Data.iloc[:, 0:3] #inputs
y = Data.iloc[:, 3:] #outputs
test_size = 0.1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

# Creating & Training the Model
model = Sequential()
model.add(Dense(units=3, activation='relu')) # Input Layer
model.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01))) # Hidden Layer
model.add(Dropout(0.1))
model.add(Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01))) # Hidden Layer
model.add(Dropout(0.1))
model.add(Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.01))) # Hidden Layer
model.add(Dropout(0.1))
model.add(Dense(units=4, activation='relu')) # Output Layer
model.compile(loss='mean_squared_error', optimizer='adam')
train = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))

# Saving & Reloading the Model
model.save('FNNO_Anes')
model = load_model('FNNO_Anes')

# Evaluating the Model
loss = model.evaluate(x_test, y_test)
print("Loss:", loss)

train_loss = train.history['loss']
test_loss = train.history['val_loss']
epochs = range(1, len(train_loss) + 1)
plt.figure()
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, test_loss, label='Test Loss')
plt.title('Training vs Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Making Predictions
y_hat = model.predict(x_test)
Columns = ['x_propo_1', 'x_propo_4', 'x_remi_1', 'x_remi_4']
y_hat = pd.DataFrame(y_hat, columns=Columns)

#%% Data Extraction
BIS = x_test.iloc[:, 2]
x1p, x4p, x1r, x4r = y_test['x_propo_1'], y_test['x_propo_4'], y_test['x_remi_1'], y_test['x_remi_4']
x1p_hat, x4p_hat, x1r_hat, x4r_hat = y_hat['x_propo_1'], y_hat['x_propo_4'], y_hat['x_remi_1'], y_hat['x_remi_4']

# Picking a Test Patient
patient = int(input(f"Enter a number from 0 to {test_size*1000-1}: "))
x1p = x1p.iloc[patient*900:(patient+1)*900]
x4p = x4p.iloc[patient*900:(patient+1)*900]
x1r = x1r.iloc[patient*900:(patient+1)*900]
x4r = x4r.iloc[patient*900:(patient+1)*900]
BIS = BIS.iloc[patient*900:(patient+1)*900]

# Estimated States
x1p_hat = x1p_hat.iloc[patient*900:(patient+1)*900]
x4p_hat = x4p_hat.iloc[patient*900:(patient+1)*900]
x1r_hat = x1r_hat.iloc[patient*900:(patient+1)*900]
x4r_hat = x4r_hat.iloc[patient*900:(patient+1)*900]

#%% Plotting the Results
Time = range(len(BIS))

# States fig
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(range(len(x1p)), x1p_hat, '--', linewidth=1.5)
axs[0, 0].plot(range(len(x1p)), x1p, linewidth=1.5)
axs[0, 0].legend(['x1p Estimated', 'x1p Measured'])
axs[0, 0].grid(True)

axs[0, 1].plot(range(len(x4p)), x4p_hat, '--', linewidth=1.5)
axs[0, 1].plot(range(len(x4p)), x4p, linewidth=1.5)
axs[0, 1].legend(['x4p Estimated', 'x4p Measured'])
axs[0, 1].grid(True)

axs[1, 0].plot(range(len(x1r)), x1r_hat, '--', linewidth=1.5)
axs[1, 0].plot(range(len(x1r)), x1r, linewidth=1.5)
axs[1, 0].legend(['x1r Estimated', 'x1r Measured'])
axs[1, 0].grid(True)

axs[1, 1].plot(range(len(x4r)), x4r_hat, '--', linewidth=1.5)
axs[1, 1].plot(range(len(x4r)), x4r, linewidth=1.5)
axs[1, 1].legend(['x4r Estimated', 'x4r Measured'])
axs[1, 1].grid(True)

plt.show()