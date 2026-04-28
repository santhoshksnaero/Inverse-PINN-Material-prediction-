import sys
sys.path.append(r'E:\Stark_Packages')
import deepxde as dde
import numpy as np


alpha = dde.Variable(0.1)

def heat_equation(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    # The AI will adjust 'alpha' to make this zero
    return dy_t - alpha * dy_xx

# 2. Geometry and Time
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 3. Generate "Sensor Data" (Simulating real-world measurements)
def real_solution(x, t):
    return np.exp(-t) * np.cos(np.pi * x / 2)

# Create 200 "Sensor" points
observe_x = np.random.uniform(-1, 1, (200, 1))
observe_t = np.random.uniform(0, 1, (200, 1))
observe_y = real_solution(observe_x, observe_t)
observe_points = np.hstack((observe_x, observe_t))

import matplotlib.pyplot as plt 
# Sort indices based on x values
idx = observe_x.argsort(axis=0).flatten()

# Plot using sorted indices
plt.plot(observe_x[idx], observe_y[idx], label="Observed Trend", color='blue', linewidth=2)
plt.scatter(observe_x, observe_y, label="Sensor Data", color='red', s=10, alpha=0.5)
plt.xlabel("Position (x)")
plt.ylabel("Temperature")
plt.title("Random Sensor Observations")
plt.show()


observe_data = dde.icbc.PointSetBC(observe_points, observe_y, component=0)

# 4. The Model
data = dde.data.TimePDE(
    geomtime, 
    heat_equation, 
    [observe_data], 
    num_domain=400, 
    num_boundary=20, 
    anchors=observe_points
)

net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)


model.compile("adam", lr=0.001, external_trainable_variables=alpha)
variable_callback = dde.callbacks.VariableValue(alpha, period=100)
model.train(iterations=2000, callbacks=[variable_callback])


alpha_value = dde.backend.to_numpy(alpha)
print(f" Discovered Alpha = {alpha_value}")



