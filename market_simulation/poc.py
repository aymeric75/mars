import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

# -----------------------
# 1. Simulate trajectories
# -----------------------

n_traces = 50
n_steps = 120
s0 = 300
q0 = 40          # initial inventory (number of sub-orders)
impact = 0.8
decay = 0.05
noise = 0.5

prices = np.zeros((n_traces, n_steps))
inventory = np.zeros((n_traces, n_steps))

for i in range(n_traces):

    p = np.zeros(n_steps)
    q = np.zeros(n_steps)

    p[0] = s0
    q[0] = q0

    for t in range(1, n_steps):

        # inventory decreases until empty
        q[t] = max(q[t-1] - 1, 0)

        if q[t-1] > 0:
            p[t] = p[t-1] + impact + np.random.randn()*noise
        else:
            p[t] = p[t-1] - decay*(p[t-1]-s0) + np.random.randn()*noise

    prices[i] = p
    inventory[i] = q

# -----------------------
# 2. Prepare data for SINDy
# -----------------------
X = [prices[i].reshape(-1,1) for i in range(n_traces)]
U = [inventory[i].reshape(-1,1) for i in range(n_traces)]

dt = 1.0

model = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=2),
    optimizer=ps.STLSQ(threshold=1e-3)
)

model.fit(X, u=U, t=dt)

print("Discovered equation:")
model.print()

# -----------------------
# 4. Plot trajectories
# -----------------------

plt.figure(figsize=(8,5))
for p in prices:
    plt.plot(p, alpha=0.4)

plt.title("Simulated price trajectories")
plt.xlabel("time")
plt.ylabel("price")
plt.title("Toy stock-price trajectories under a meta order")
plt.legend()
plt.tight_layout()
plt.savefig("meta_order_traces.png", dpi=150)
#plt.show()
