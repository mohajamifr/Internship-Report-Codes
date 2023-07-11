import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, BatchNormalization, LSTM, GRU, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Loading the Data
Data=pd.read_csv("Data.csv")
Data = Data[['u_propo', 'u_remi', 'BIS', 'x_propo_1', 'x_propo_4', 'x_remi_1', 'x_remi_4']]

# Creating the Training/Test Sets
x = Data.iloc[:, 0:3] #inputs
y = Data.iloc[:, 3:] #outputs
x = x.values.reshape(1000, 900, x.shape[1])
y = y.values.reshape(1000, 900, y.shape[1])
test_size = 0.1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

# Creating & Training the Model
model = Sequential()
model.add(GRU(units=256, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(GRU(units=128, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(GRU(units=64, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(GRU(units=32, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(units=4, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
train = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))

# Saving & Reloading the Model
model.save('RNNO_Anes')
model = load_model('RNNO_Anes')

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

# Picking a Test Patient
patient = int(input(f"Enter a number from 0 to {test_size*1000-1}: "))
x1p = y_test[patient, :, 0]
x4p = y_test[patient, :, 1]
x1r = y_test[patient, :, 2]
x4r = y_test[patient, :, 3]
BIS = x_test[patient, :, 2]

# Estimated States
x1p_hat = y_hat[patient, :, 0]
x4p_hat = y_hat[patient, :, 1]
x1r_hat = y_hat[patient, :, 2]
x4r_hat = y_hat[patient, :, 3]

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

# Numerical Analysis
x1P, x4P = x1p_hat[len(x1p_hat)//3:], x4p_hat[len(x4p_hat)//3:]
x1R, x4R = x1r_hat[len(x1r_hat)//3:], x4r_hat[len(x4r_hat)//3:]

mean_x1P, std_x1P, mean_x1R, std_x1R = x1P.mean(), x1P.std(), x1R.mean(), x1R.std()
mean_x4P, std_x4P, mean_x4R, std_x4R = x4P.mean(), x4P.std(), x4R.mean(), x4R.std()

mean = [mean_x1P, mean_x4P, mean_x1R, mean_x4R]
std = [std_x1P, std_x4P, std_x1R, std_x4R]
true_value = [x1p[len(x1p)//3:].mean(), x4p[len(x4p)//3:].mean(), x1r[len(x1r)//3:].mean(), x4r[len(x4r)//3:].mean()]

# Comparison fig
plt.figure()
plt.bar(range(len(mean)), mean, yerr=std, alpha=0.5, ecolor='black', capsize=5, label='Estimated Values')
plt.errorbar(range(len(mean)), true_value, yerr=np.std(true_value), fmt='o', ecolor='red', label='True Values')
plt.xticks(range(len(mean)), ['x1p', 'x4p', 'x1r', 'x4r'])
plt.ylabel('Value')
plt.legend()

plt.show()