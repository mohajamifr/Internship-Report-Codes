# %% imports

# Loading Libraries
import casadi as cas  # to be used for numerical optimization
import numpy as np  # to perform mathematical operations
import pandas as pd  # to be used as a data manipulation tool
from scipy.integrate import solve_ivp  # to solve ordinary differential equations
from matplotlib import pyplot as plt  # to plot figures
import vitaldb as vdb  # to load patients' data online
import python_anesthesia_simulator as pas

# Loading Patients' Data
Patient_Info = pd.read_csv("./Data/parameters.csv")

# %% Moving Horizon Estimator


class MHE():
    def __init__(self, Case_ID, N_MHE: int = 20, model: list = ['Eleveld']*2):
        self.Case_ID = Case_ID  # patient case number
        self.N_MHE = N_MHE  # estimation horizon
        self.ts = 1  # sampling time in ms
        self.N_samp = 2  # number of samples
        self.T = self.N_samp  # sampling period

        age, height, weight, sex = int(Patient_Info['age'][self.Case_ID]), float(Patient_Info['height'][self.Case_ID]), float(
            Patient_Info['weight'][self.Case_ID]), Patient_Info['gender'][self.Case_ID]
        if sex == "M":
            sex = 1  # Male (M)
        else:
            sex = 0  # Female (F)
        self.Patient_Char = [age, height, weight, sex]

        Patient_Variables = pd.read_csv(f"./Data/simu_{self.Case_ID}.csv").values

        BIS, Propo_rate, Remi_rate = Patient_Variables[:, 2], Patient_Variables[:, 6], Patient_Variables[:, 7]
        self.Patient_Var = [BIS, Propo_rate, Remi_rate]

        xp1, xp2, xp3, xp4, xr1, xr2, xr3, xr4 = Patient_Variables[:, 9], Patient_Variables[:, 10], Patient_Variables[:, 11], Patient_Variables[:, 12], Patient_Variables[:, 15], Patient_Variables[:, 16], Patient_Variables[:, 17], Patient_Variables[:, 18]
        self.states = [xp1, xp2, xp3, xp4, xr1, xr2, xr3, xr4]

        Patient = pas.Patient(patient_characteristic=self.Patient_Char, ts=self.ts,
                              model_propo=model[0], model_remi=model[1], co_update=True)
        c50p, c50r, gamma, beta, E0, Emax = Patient.hill_param
        self.Hill_Par = [c50p, c50r, gamma, beta, E0, Emax]

        A_p = Patient.propo_pk.continuous_sys.A[:4, :4]
        A_r = Patient.remi_pk.continuous_sys.A[:4, :4]
        B_p = Patient.propo_pk.continuous_sys.B[:4]
        B_r = Patient.remi_pk.continuous_sys.B[:4]

        self.A = np.block([[A_p, np.zeros((4, 4))], [np.zeros((4, 4)), A_r]])
        self.B = np.block([[B_p, np.zeros((4, 1))], [np.zeros((4, 1)), B_r]])

        self.BIS, self.y_measurement, self.Time, self.u_p, self.u_r = self.BIS_Real()
        self.u = np.block([self.u_p, self.u_r])

    def BIS_Real(self):
        Patient_Var = self.Patient_Var
        BIS = Patient_Var[0]
        BIS = BIS.T
        y_measurements = pd.DataFrame(BIS)

        Propo_rate = Patient_Var[1]
        u_p = pd.DataFrame(Propo_rate)

        Remi_rate = Patient_Var[2]
        u_r = pd.DataFrame(Remi_rate)

        # MHE problem formulation
        Time = [x for x in range(1, len(u_r)+1)]
        N_samp = self.N_samp  # number of samples

        # Reduce the number of elements in the measured data
        Time = np.array(Time)[0::N_samp]
        BIS = BIS[0::N_samp]  # measured BIS
        y_measurements = y_measurements.iloc[0::N_samp]
        u_p, u_r = u_p.iloc[0::N_samp], u_r.iloc[0::N_samp]

        # BIS filtering
        window_size = 5
        y_measurements.reset_index(drop=True, inplace=True)
        y_measurements = y_measurements.rolling(window=window_size, center=True,
                                                min_periods=1).mean().values  # filtered BIS

        E0 = np.mean(y_measurements[:10])
        Emax = E0
        self.Hill_Par[-1] = Emax
        self.Hill_Par[-2] = E0
        return BIS, y_measurements, Time, u_p, u_r

    def BIS_Estimated(self):
        # Model simulation with the real input

        # Model Simulation
        xx0 = np.zeros([8, 1])

        def model(t, x, u, A, B, T):
            uu = np.array([np.interp(t, T, u[:, i]) for i in range(2)])
            return (A @ x + B @ uu).flatten()
        ODE_sol = solve_ivp(lambda t, x: model(t, x, self.u, self.A, self.B, self.Time),
                            [self.Time[0], self.Time[-1]], xx0.flatten(), method='RK45',
                            t_eval=self.Time, rtol=1e-6)
        t = ODE_sol.t
        y = ODE_sol.y.T

        xx1 = np.interp(self.Time, t, y[:, 0])
        xx2 = np.interp(self.Time, t, y[:, 1])
        xx3 = np.interp(self.Time, t, y[:, 2])
        xx4 = np.interp(self.Time, t, y[:, 3])

        xx5 = np.interp(self.Time, t, y[:, 4])
        xx6 = np.interp(self.Time, t, y[:, 5])
        xx7 = np.interp(self.Time, t, y[:, 6])
        xx8 = np.interp(self.Time, t, y[:, 7])

        # Data Extension
        N_samp = self.N_samp
        T = self.T
        c50p, c50r, gamma, beta, E0, Emax = self.Hill_Par

        N_ex = self.N_MHE//1
        tt1 = np.arange(0, N_samp*(N_ex), N_samp)
        Time = np.concatenate((tt1, self.Time+tt1[-1]-self.Time[0]+T))
        u_r = np.vstack((np.zeros([N_ex, 1]), self.u[:, [1]]))
        u_p = np.vstack((np.zeros([N_ex, 1]), self.u[:, [0]]))
        xx1 = np.vstack((np.zeros([N_ex, 1]), xx1.reshape(-1, 1))).T
        xx2 = np.vstack((np.zeros([N_ex, 1]), xx2.reshape(-1, 1))).T
        xx3 = np.vstack((np.zeros([N_ex, 1]), xx3.reshape(-1, 1))).T
        xx4 = np.vstack((np.zeros([N_ex, 1]), xx4.reshape(-1, 1))).T
        xx5 = np.vstack((np.zeros([N_ex, 1]), xx5.reshape(-1, 1))).T
        xx6 = np.vstack((np.zeros([N_ex, 1]), xx6.reshape(-1, 1))).T
        xx7 = np.vstack((np.zeros([N_ex, 1]), xx7.reshape(-1, 1))).T
        xx8 = np.vstack((np.zeros([N_ex, 1]), xx8.reshape(-1, 1))).T
        xx = [xx1, xx2, xx3, xx4, xx5, xx6, xx7, xx8]
        y_measurements = np.vstack((E0*np.ones([N_ex, 1]), np.reshape(self.y_measurement.T, (-1, 1))))  # filtered BIS
        BIS = np.vstack((E0*np.ones([N_ex, 1]), self.BIS.reshape(-1, 1)))  # measured BIS
        UUp, UUr = xx4/c50p, xx8/c50r
        theta = np.divide(UUp, UUp + UUr + 1e-6)
        Uu = np.divide(UUp + UUr, 1 - beta*theta + beta*theta**2)
        BIS_mod = E0 - np.divide(Emax*(Uu**gamma), 1 + Uu**gamma)  # estimated BIS
        return BIS_mod, Time, u_p, u_r, BIS, y_measurements, xx

    def Plot_Comp(self):
        BIS_mod, Time, u_p, u_r, BIS, y_measurements, xx = self.BIS_Estimated()

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(Time, u_p, label='Propo')
        axs[0].plot(Time, u_r, label='Remi')
        axs[0].legend()
        axs[0].set_title('Drug Injection Rates (input)')
        axs[0].grid(True)

        axs[1].plot(Time, BIS, label='BIS Measured')
        axs[1].plot(Time, BIS_mod.real.flatten(),
                    linewidth=2, label='BIS Model')
        axs[1].legend()
        axs[1].set_title('BIS (output)')
        axs[1].grid(True)

        plt.show()

    def OPT_Prob(self):

        # Define the model
        # ----------------------------------------
        x1, x2, x3, x4, x5, x6, x7, x8, C50p, C50r, Gamma = cas.SX.sym('x1'), cas.SX.sym('x2'), cas.SX.sym('x3'), cas.SX.sym('x4'), cas.SX.sym(
            'x5'), cas.SX.sym('x6'), cas.SX.sym('x7'), cas.SX.sym('x8'), cas.SX.sym('C50p'), cas.SX.sym('C50r'), cas.SX.sym('Gamma')

        states = [x1, x2, x3, x4, x5, x6, x7, x8, C50p, C50r, Gamma]
        n_states = len(states)
        states = cas.vertcat(*states)
        self.States = [states, n_states]

        vp, vr = cas.SX.sym('vp'), cas.SX.sym('vr')
        controls = [vp, vr]
        n_controls = len(controls)
        controls = cas.vertcat(*controls)
        self.Inputs = [controls, n_controls]

        Beta = self.Hill_Par[3]
        E0 = self.Hill_Par[4]
        Emax = self.Hill_Par[5]
        rhs = self.A @ states[:8] + self.B @ controls

        # Propofol PK model
        rhs = cas.vertcat(rhs, 0, 0, 0)
        # Linear PK mapping function f(x, u)
        f = cas.Function('f', [states, controls], [rhs])
        Up, Ur = x4/C50p, x8/C50r
        theta = Up/(Up+Ur+1e-6)
        UU = (Up+Ur)/(1-Beta*theta+Beta*theta**2)
        measurement_rhs = E0-Emax*(UU**Gamma)/(1+UU**Gamma)
        # Measurement model
        h = cas.Function('h', [states], [measurement_rhs])

        # Define the objective function
        # ----------------------------------------

        X = cas.SX.sym('x', n_states, (self.N_MHE+1))  # states
        P = cas.SX.sym('P', 1, n_controls*self.N_MHE + (self.N_MHE + 1) +
                       (self.N_MHE + 1)*n_states + 1 + 1)  # parameters

        Q = P[-1]  # weighting matrices (output)  y_tilde - y
        R = np.eye(n_states - 3)
        R[1, 1], R[2, 2], R[5, 5], R[6, 6] = 550, 550, 50, 750

        theta_1_c50p = 1
        theta_2_c50p = 100
        theta_3_c50p = 300
        theta_4_c50p = 0.005

        theta_1_c50r = 0.01
        theta_2_c50r = 1
        theta_3_c50r = 300
        theta_4_c50r = 0.005

        theta_1_gamma = 1
        theta_2_gamma = 50
        theta_3_gamma = 300
        theta_4_gamma = 0.005

        obj = 0  # objective function
        N_it = P[:, -1]

        for k in range(0, self.N_MHE+1):
            st = X[:, k]
            h_x = h(st)
            y_tilde = P[:, k]
            obj += (y_tilde - h_x).T * Q * (y_tilde - h_x)  # Calculate obj

        for k in range(0, self.N_MHE - 1):
            st1 = P[3*self.N_MHE + k+1: 1 + 3*self.N_MHE + k + n_states*(self.N_MHE) - 7: self.N_MHE + 1]
            con1 = cas.horzcat(P[:, self.N_MHE + k+1], P[:, (1 + self.N_MHE) + self.N_MHE + k])
            f_value1 = f(st1, con1)
            st1_next = st1 + (self.T * f_value1.T)

            obj += (X[0:8, k].T - st1_next[0, 0:8]
                    ) @ R @ (X[0:8, k].T - st1_next[0, 0:8]).T
            obj += (X[8, k].T - st1_next[0, 8])**2 * (theta_1_c50p + (theta_2_c50p) *
                                                      np.exp(-theta_3_c50p * np.exp(-theta_4_c50p*N_it)))
            obj += (X[9, k].T - st1_next[0, 9])**2 * (theta_1_c50r + (theta_2_c50r) *
                                                      np.exp(-theta_3_c50r * np.exp(-theta_4_c50r*N_it)))
            obj += (X[10, k].T - st1_next[0, 10])**2 * (theta_1_gamma + (theta_2_gamma) *
                                                        np.exp(-theta_3_gamma * np.exp(-theta_4_gamma*N_it)))

        # Define the constraints function
        # ----------------------------------------

        x1_max = 30
        x2_max = 30
        x3_max = 30
        x4_max = 30
        x5_max = 30
        x6_max = 30
        x7_max = 30
        x8_max = 30
        x1_min = x2_min = x3_min = x4_min = x5_min = x6_min = x7_min = x8_min = 0
        x9_max, x9_min, x10_max, x10_min, x11_max, x11_min = 15, 1e-3, 60, 1e-3, 16, 1e-3
        g = []  # constraints vector
        for k in range(0, self.N_MHE):
            st = X[:, k]
            con = cas.horzcat(P[:, 1 + self.N_MHE + k], P[:, (1 + self.N_MHE) + self.N_MHE + k])
            st_next = X[:, k + 1]
            f_value = f(st, con)
            st_next_euler = st + (self.T * f_value)
            g = cas.vertcat(g, st_next - st_next_euler)  # compute constraints

        # Make the decision variable one column vector
        OPT_variables = cas.horzcat(cas.reshape(X, n_states*(self.N_MHE+1), 1))
        nlp_mhe = {'f':  obj, 'x': OPT_variables, 'g': g, 'p': P}
        opts = {'ipopt.max_iter': 2000,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-6}
        solver = cas.nlpsol('solver', 'ipopt', nlp_mhe, opts)

        lbg = np.zeros([1, n_states*self.N_MHE])
        ubg = np.zeros([1, n_states*self.N_MHE])

        lbx = np.zeros([n_states*(self.N_MHE+1), 1])
        ubx = np.zeros([n_states*(self.N_MHE+1), 1])
        lbx[0::n_states] = x1_min
        ubx[0::n_states] = x1_max
        lbx[1::n_states] = x2_min
        ubx[1::n_states] = x2_max
        lbx[2::n_states] = x3_min
        ubx[2::n_states] = x3_max
        lbx[3::n_states] = x4_min
        ubx[3::n_states] = x4_max
        lbx[4::n_states] = x5_min
        ubx[4::n_states] = x5_max
        lbx[5::n_states] = x6_min
        ubx[5::n_states] = x6_max
        lbx[6::n_states] = x7_min
        ubx[6::n_states] = x7_max
        lbx[7::n_states] = x8_min
        ubx[7::n_states] = x8_max
        lbx[8::n_states] = x9_min
        ubx[8::n_states] = x9_max
        lbx[9::n_states] = x10_min
        ubx[9::n_states] = x10_max
        lbx[10::n_states] = x11_min
        ubx[10::n_states] = x11_max

        self.f = f
        self.n_states = n_states

        return solver, lbg, ubg, lbx, ubx

###########################################################
######## ALL OF THE ABOVE IS JUST A PROBLEM SET UP ########
###########################################################

###########################################################
############# MHE Simulation loop starts here #############
###########################################################

    def BIS_MHE(self):
        N_MHE = self.N_MHE

        c50p, c50r, gamma, beta, E0, Emax = self.Hill_Par
        BIS_Estimated = self.BIS_Estimated()
        u_p, u_r, y_measurements = BIS_Estimated[2], BIS_Estimated[3], BIS_Estimated[5]
        solver, lbg, ubg, lbx, ubx = self.OPT_Prob()

        X_estimate = []  # contains the MHE estimate of the states

        X0 = np.zeros([N_MHE + 1, self.n_states])
        X0[:, : self.n_states - 3] = np.zeros([N_MHE + 1, self.n_states - 3])
        X0[:, 8] = c50p*np.ones([N_MHE + 1])
        X0[:, 9] = c50r*np.ones([N_MHE + 1])
        X0[:, 10] = gamma*np.ones([N_MHE + 1])

        OBJ = []

        # Initialize the previous estimated state
        X_sol = np.zeros([N_MHE + 1, self.n_states])
        X_sol[:, 8] = c50p*np.ones([N_MHE + 1])
        X_sol[:, 9] = c50r*np.ones([N_MHE + 1])
        X_sol[:, 10] = gamma*np.ones([N_MHE + 1])

        for k in range(0, len(y_measurements) - N_MHE):

            p = np.hstack((y_measurements[k:k+N_MHE+1, :].T, u_p[k:k+N_MHE, :].T, u_r[k:k+N_MHE, :].T,
                          np.reshape(X_sol.T, (1, (N_MHE+1)*self.n_states)), np.array([k+1]).reshape(-1, 1), np.array([[0.005]])))
            if k > 25*60/2:
                p = np.hstack((y_measurements[k:k+N_MHE+1, :].T, u_p[k:k+N_MHE, :].T, u_r[k:k+N_MHE, :].T,
                               np.reshape(X_sol.T, (1, (N_MHE+1)*self.n_states)), np.array([k+1]).reshape(-1, 1), np.array([[0]])))
            x0 = X0.reshape(((N_MHE+1)*self.n_states, 1))
            sol = solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=lbg.T, ubg=ubg.T)
            X_sol = sol['x'][0:self.n_states * (N_MHE+1)].T.reshape((self.n_states, N_MHE+1)).T
            X_estimate.append(X_sol[N_MHE, :])
            X0 = np.vstack((X_sol[1:, :], X_sol[-1, :] + self.f(X_sol[-1, :],
                           [u_p[k+N_MHE-1, :].T, u_r[k+N_MHE-1, :].T]).T))
            OBJ.append(sol['f'])

        X_estimate = np.array(X_estimate).squeeze()
        self.cost = OBJ
        UUp, UUr = X_estimate[:, 3]/X_estimate[:, 8], X_estimate[:, 7]/X_estimate[:, 9]
        theta = UUp/(UUp+UUr)
        U = (UUp + UUr)/(1 - beta * theta + beta*theta**2)
        self.BIS_estimated = E0 - Emax * (U**X_estimate[:, 10])/(1 + U**X_estimate[:, 10])
        self.X_estimate = X_estimate
        return self.BIS_estimated, self.X_estimate

    def Plot_EstBIS(self):
        N_MHE = self.N_MHE
        BIS_mod, Time, _, _, BIS, _, _ = self.BIS_Estimated()

        plt.figure()
        plt.plot(Time, BIS, linewidth=1.5, label='BIS measured')
        plt.plot(Time, BIS_mod[0], linewidth=1.5, label='BIS nominal model')
        plt.plot(Time[N_MHE:], self.BIS_estimated, '--',
                 linewidth=1.5, label='BIS estimated')
        plt.legend()
        plt.show()

    def Plot_Params(self):
        N_MHE = self.N_MHE
        Time = self.Time
        C50p, C50r, Gamma = float(Patient_Info['c50p'][self.Case_ID]), float(Patient_Info['c50r'][self.Case_ID]), float(Patient_Info['gamma'][self.Case_ID])

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(Time, self.X_estimate[:, 8], '--', linewidth=1.5)
        axs[0].plot(Time, np.repeat(C50p, len(Time)))
        axs[0].legend(['C50p Estimated', 'C50p Real'])
        axs[0].grid(True)

        axs[1].plot(Time[:], self.X_estimate[:, 9], '--', linewidth=1.5)
        axs[1].plot(Time, np.repeat(C50r, len(Time)))
        axs[1].legend(['C50r Estimated', 'C50r Real'])
        axs[1].grid(True)

        axs[2].plot(Time, self.X_estimate[:, 10], '--', linewidth=1.5)
        axs[2].plot(Time, np.repeat(Gamma, len(Time)))
        axs[2].legend(['Gamma Estimated', 'Gamma Real'])
        axs[2].grid(True)

        actual = np.array([C50p, C50r, Gamma])
        estimated = self.X_estimate[-1:, 8:11].flatten()
        error = abs(actual-estimated)/actual
        column_titles = ['c50p', 'c50r', 'gamma']
        RError = pd.DataFrame({'Real': actual, 'Estimated': estimated, '% Error': error}, index=column_titles)
        RError.to_excel("PDMHE_RE.xlsx")

        plt.show()

    def Plot_States(self):
        Time = self.Time
        N_MHE = self.N_MHE
        fig, axs = plt.subplots(8, 1)
        for i in range(8):
            axs[i].plot(Time, self.X_estimate[:, i], '--', linewidth=1.5)
            axs[i].plot(Time, self.states[i][0::self.N_samp])
            axs[i].legend([f'x{i+1} Estimated', f'x{i+1} Real'])
            axs[i].grid(True)

        # Root Mean Squared Error
        rmse=[]
        rmse_perc=[]
        for i in range(8):
            actual = np.array(self.states).T[len(Time)*self.N_samp//3::self.N_samp, i]
            estimated = self.X_estimate[len(Time)//3:, i]
            RMSE = np.sqrt(np.mean((actual-estimated)**2, axis=0))
            RMSE_perc = RMSE/np.mean(actual, axis=0)
            rmse.append(RMSE)
            rmse_perc.append(RMSE_perc)
         
        column_titles = ['x1p', 'x2p', 'x3p', 'x4p', 'x1r', 'x2r', 'x3r', 'x4r']
        RMSE = pd.DataFrame({'RMSE': rmse, '%': rmse_perc}, index=column_titles)
        RMSE.to_excel("SMHE_RMSE.xlsx")

        plt.show()

    def Save_Results(self):
        # creat a date frame with the tthe results of the MHE
        BIS_mod, Time, _, _, BIS, _, _ = self.BIS_Estimated()
        Time = self.Time/60
        N_MHE = self.N_MHE
        dataframe = pd.DataFrame()
        dataframe['Time'] = Time
        dataframe['BIS_measured'] = self.BIS
        dataframe['BIS_model'] = BIS_mod[0][:-N_MHE]
        dataframe['BIS_MHE'] = self.BIS_estimated
        dataframe['u_propo'] = self.u_p
        dataframe['u_remi'] = self.u_r
        for i in range(11):
            dataframe[f'x_{i}'] = self.X_estimate[:, i]
        dataframe.to_csv(f'./results/MHE_{self.Case_ID:04d}.csv', index=False)

if __name__ == "__main__":
    N_MHE = 20
    Case_ID = 907
    MHE_instance = MHE(Case_ID=Case_ID, N_MHE=N_MHE)
    MHE_instance.BIS_MHE()
    MHE_instance.Plot_Comp()
    MHE_instance.Plot_EstBIS()
    MHE_instance.Plot_States()
    MHE_instance.Plot_Params()