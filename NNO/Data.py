import pandas as pd
import numpy as np

PC = pd.read_csv("./Data/parameters.csv") # Patient Characteristics
PC = pd.DataFrame(np.repeat(PC.values, 900, axis=0), columns = PC.columns).iloc[:900000, 1:]
PVS = []
for i in range(1000):
    PV = pd.read_csv(f"./Data/simu_{i}.csv") # Patient Variables
    PVS.append(PV)
PV = pd.concat(PVS).iloc[:, 1:]
PC.reset_index(drop=True, inplace=True)
PV.reset_index(drop=True, inplace=True)
Data = pd.concat([PC, PV], axis=1) # Data set containing Patient Characteristics & Variables
Data = Data[['age', 'height', 'weight', 'gender', 'u_propo', 'u_remi', 'BIS', 'x_propo_1', 'x_propo_2', 'x_propo_3', 'x_propo_4', 'x_remi_1', 'x_remi_2', 'x_remi_3', 'x_remi_4', 'c50p', 'c50r', 'gamma']]
Data.to_csv("Data.csv", index=False)
