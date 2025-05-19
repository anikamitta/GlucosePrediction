
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import cvxpy as cp

# Define Hovorka Parameters
p = {
    'EGP0': 0.0161, 
    'F01': 0.0097,
    'Vg': 0.16,
    'V_I': 0.12,
    'k_e': 0.138,
    'k_a1': 0.006,
    'k_a2': 0.06,
    'k_a3': 0.03,
    'S_I': 51.2e-4,
    'BW': 70.0
}

# Dummy LSTM Predictor (CPU-compatible)
def dummy_lstm_predictor(glucose_hist, insulin_hist, meal_hist, horizon=10):
    last_glucose = glucose_hist[-1]
    trend = (glucose_hist[-1] - glucose_hist[-2]) if len(glucose_hist) >= 2 else 0
    return np.array([last_glucose + i * trend * 0.5 for i in range(1, horizon + 1)])

# Hovorka ODE Function
def hovorka_model(t, y, insulin):
    Q1, Q2, I, x1, x2, x3 = y
    Vg, BW = p['Vg'], p['BW']
    G = Q1 / (Vg * BW)

    dQ1 = -p['F01'] * BW - x1 * Q1 + 0.01 * (Q2 - Q1) + p['EGP0'] * (1 - x3)
    dQ2 = x1 * Q1 - 0.01 * (Q2 - Q1)
    dI = -p['k_e'] * I + insulin
    dx1 = -p['k_a1'] * x1 + p['S_I'] * I
    dx2 = -p['k_a2'] * x2 + p['S_I'] * I
    dx3 = -p['k_a3'] * x3 + p['S_I'] * I

    return [dQ1, dQ2, dI, dx1, dx2, dx3]

# Initialize Simulation
time_horizon = 10
total_steps = 150
state = [11.0 * p['Vg'] * p['BW']] * 2 + [0.0] + [0.0]*3
glucose_log, insulin_log = [], []

# History for predictor input
glucose_hist = [11.0]*10
insulin_hist = [0.0]*10
meal_hist = [0.0]*10

# Simulation Loop
for t in range(total_steps):
    G_current = state[0] / (p['Vg'] * p['BW'])
    glucose_log.append(G_current)

    # Use dummy predictor (replace with real LSTM later)
    predicted_glucose = dummy_lstm_predictor(glucose_hist, insulin_hist, meal_hist, time_horizon)

    # MPC Optimization
    u = cp.Variable(time_horizon)
    insulin_bounds = (0.0, 1.0)
    G_target = 6.0
    cost = 0
    constraints = []

    for i in range(time_horizon):
        cost += cp.square(predicted_glucose[i] - G_target) + 0.01 * cp.square(u[i])
        constraints += [u[i] >= insulin_bounds[0], u[i] <= insulin_bounds[1]]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    insulin_dose = u.value[0] if u.value is not None else 0.0
    insulin_log.append(insulin_dose)

    # Step Hovorka model
    sol = solve_ivp(lambda t_, y_: hovorka_model(t_, y_, insulin_dose),
                    [0, 1], state, method='RK45')
    state = sol.y[:, -1].tolist()

    # Update history
    glucose_hist.append(G_current)
    insulin_hist.append(insulin_dose)
    meal_hist.append(0.0)
    glucose_hist.pop(0)
    insulin_hist.pop(0)
    meal_hist.pop(0)

# Plot Results
time = np.arange(total_steps)
plt.figure()
plt.plot(time, glucose_log, label='Glucose (mmol/L)')
plt.axhline(G_target, color='gray', linestyle='--', label='Target')
plt.xlabel("Time (min)")
plt.ylabel("Glucose")
plt.title("MPC with Dummy LSTM + Hovorka Model")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(time, insulin_log, label='Insulin (U/min)')
plt.xlabel("Time (min)")
plt.ylabel("Insulin Dose")
plt.title("Insulin Delivery via MPC")
plt.grid()
plt.show()