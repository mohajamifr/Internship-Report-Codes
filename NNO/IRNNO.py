import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as cas
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import MHE_modified as MHE

# Loading the Data
Data=pd.read_csv("Data.csv")
Data = Data[['u_propo', 'u_remi', 'BIS', 'x_propo_1', 'x_propo_4', 'x_remi_1', 'x_remi_4', 'c50p', 'c50r', 'gamma']]

# Creating the Training/Test Sets
x = Data.iloc[:, 0:3].values #inputs
y = Data.iloc[:, 3:].values #outputs
x = x.reshape(1000, 900, x.shape[1])
y = y.reshape(1000, 900, y.shape[1])
test_size = 0.1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

# %% Estimating the States
# Creating, Compiling and Training the Model
model = Sequential()
model.add(GRU(units=256, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(GRU(units=128, activation='relu', return_sequences=True))
model.add(Dense(units=4, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
train = model.fit(x_train, y_train[:, :, :4], epochs=100, batch_size=128, validation_data=(x_test, y_test[:, :, :4]), callbacks=[early_stopping])

# Saving/Loading the Model
model.save('CNNO1_Anes')
model = load_model('CNNO1_Anes')

# Evaluating the Model
loss = model.evaluate(x_test, y_test[:, :, :4])
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
y_hat = model.predict(x_test) # Estimated States

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

#%% Estimating the PD Parameters

# Defining the Variables to be used in the objective function
c50p = cas.SX.sym('c50p')
c50r = cas.SX.sym('c50r')
gamma = cas.SX.sym('gamma')
y = cas.SX.sym('y')

# Defining the constraints
constr = cas.vertcat(c50p, c50r, gamma) >= 0

# Defining the objective function
obj = 0
Time=range(y_hat.shape[1])
E0, Emax = BIS[0], max(BIS)
for i in Time:
    obj += (E0-Emax*((x4p_hat[i]/c50p + x4r_hat[i]/c50r)**gamma)/(1+(x4p_hat[i]/c50p + x4r_hat[i]/c50r)**gamma)-BIS[i])**2

# Defining the optimization problem
opt_prob = {'f': obj, 'x': cas.vertcat(c50p, c50r, gamma), 'g': constr}
opts = {'ipopt.max_iter': 2000,
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6}
solver = cas.nlpsol('solver', 'ipopt', opt_prob, opts)

# Solving the optimization problem
x0 = [0, 0, 0]
C50p, C50r, Gamma = y_test[:, :, 4], y_test[:, :, 5], y_test[:, :, 6]
res = solver(x0=x0,lbx=[min(C50p.flatten()), min(C50r.flatten()), min(Gamma.flatten())], ubx=[max(C50p.flatten()), max(C50r.flatten()), max(Gamma.flatten())])

# Extract the PD parameters
x_opt = res['x']
c50p, c50r, gamma = float(x_opt[0]), float(x_opt[1]), float(x_opt[2])

#%% Estimating the BIS
BIS_real = x_test[patient, :, 2]
BIS_hat = E0-Emax*((x4p_hat/c50p + x4r_hat/c50r)**gamma)/(1+(x4p_hat/c50p + x4r_hat/c50r)**gamma)

#%% Plotting the Results

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

# BIS fig
plt.figure()
plt.plot(Time, BIS, linewidth=1.5, label='BIS Measured')
plt.plot(Time, BIS_hat, '--', linewidth=1.5, label='BIS Estimated')
plt.ylabel('% BIS')
plt.xlabel('Time')
plt.title('Measured vs Estimated BIS')
plt.legend()
plt.grid(True)

# MHE vs IRNNO BIS fig
plt.figure()
mhe = MHE.MHE(907, 20)
BIS_MHE = mhe.BIS_MHE()[0]
BIS = BIS[::2]
BIS_hat = BIS_hat[::2]
plt.plot(Time[::2], BIS, linewidth=1.5, label='BIS Measured')
plt.plot(Time[::2], BIS_hat, '--', linewidth=1.5, label='IRNNO BIS')
plt.plot(Time[::2], BIS_MHE, '--', linewidth=1.5, label='MHE BIS')
plt.ylabel('% BIS')
plt.xlabel('Time')
plt.title('Measured vs Estimated BIS')
plt.legend()
plt.grid(True)

RMSE_BIS_MHE = np.sqrt(np.mean((BIS[len(BIS)//3:]-BIS_MHE[len(BIS)//3:])**2, axis=0))
RMSE_BIS_IRNNO = np.sqrt(np.mean((BIS[len(BIS)//3:]-BIS_hat[len(BIS)//3:])**2, axis=0))
column_titles = ['MHE', 'IRNNO']
RMSE = pd.DataFrame({'RMSE': [RMSE_BIS_MHE, RMSE_BIS_IRNNO], '%RMSE': [RMSE_BIS_MHE, RMSE_BIS_IRNNO]/np.mean(BIS)}, index=column_titles)
RMSE.to_excel("BIS_RMSE.xlsx")

# PD Params fig
C50p, C50r, Gamma = C50p[patient, :], C50r[patient, :], Gamma[patient, :]
fig, axs = plt.subplots(3, 1)
axs[0].plot(Time, np.repeat(c50p, len(Time), axis=0), '--', linewidth=1.5)
axs[0].plot(Time, C50p, linewidth=1.5)
axs[0].legend(['C50p Estimated', 'C50p Real'])
axs[0].set_ylabel('C50p')
axs[0].grid(True)

axs[1].plot(Time, np.repeat(c50r, len(Time), axis=0), '--', linewidth=1.5)
axs[1].plot(Time, C50r, linewidth=1.5)
axs[1].legend(['C50r Estimated', 'C50r Real'])
axs[1].set_ylabel('C50r')
axs[1].grid(True)

axs[2].plot(Time, np.repeat(gamma, len(Time), axis=0), '--', linewidth=1.5)
axs[2].plot(Time, Gamma, linewidth=1.5)
axs[2].legend(['Gamma Estimated', 'Gamma Real'])
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Gamma')
axs[2].grid(True)

# Numerical Analysis
x1P, x4P = x1p_hat[len(x1p_hat)//3:], x4p_hat[len(x4p_hat)//3:]
x1R, x4R = x1r_hat[len(x1r_hat)//3:], x4r_hat[len(x4r_hat)//3:]

mean_x1P, std_x1P, mean_x1R, std_x1R = x1P.mean(), x1P.std(), x1R.mean(), x1R.std()
mean_x4P, std_x4P, mean_x4R, std_x4R = x4P.mean(), x4P.std(), x4R.mean(), x4R.std()
mean = [mean_x1P, mean_x4P, mean_x1R, mean_x4R]
std = [std_x1P, std_x4P, std_x1R, std_x4R]
true_value = [x1p[len(x1p)//3:].mean(), x4p[len(x4p)//3:].mean(), x1r[len(x1r)//3:].mean(), x4r[len(x4r)//3:].mean()]

# Root Mean Squared Error
actual = np.vstack([x1p[len(x1p)//3:], x4p[len(x4p)//3:], x1r[len(x1r)//3:], x4r[len(x4r)//3:]]).T
estimated = np.vstack([x1P, x4P, x1R, x4R]).T
RMSE = np.sqrt(np.mean((actual-estimated)**2, axis=0))
RMSE_perc = RMSE/np.mean(actual, axis=0)
column_titles = ['x1p', 'x4p', 'x1r', 'x4r']
RMSE = pd.DataFrame({'RMSE': RMSE, '%': RMSE_perc}, index=column_titles)
RMSE.to_excel("SRNNO_RMSE.xlsx")

# Relative Error
actual = np.array([C50p[0], C50r[0], Gamma[0]])
estimated = np.array([c50p, c50r, gamma])
error = abs(actual-estimated)/actual
column_titles = ['c50p', 'c50r', 'gamma']
RError = pd.DataFrame({'Real': actual, 'Estimated': estimated, '% Error': error}, index=column_titles)
RError.to_excel("PDRNNO_RE.xlsx")

plt.show()