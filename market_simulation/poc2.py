import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

# -----------------------
# 1. Ground truth model
# p' = a*u - b*(p-p0)
# -----------------------

n_traces = 30
n_steps = 120
dt = 1.0

p0 = 300
a = 0.6        # impact strength
b = 0.04       # resilience
noise = 0.3

prices = []
flows = []

for i in range(n_traces):

    p = np.zeros(n_steps)
    u = np.zeros(n_steps)

    p[0] = p0
    inventory = np.random.randint(30,50)

    for t in range(1,n_steps):

        trade = min(np.random.randint(1,3),inventory)
        inventory -= trade

        u[t] = trade

        dp = a*u[t] - b*(p[t-1]-p0)
        p[t] = p[t-1] + dt*dp + np.random.randn()*noise

    prices.append(p.reshape(-1,1))
    flows.append(u.reshape(-1,1))


# -----------------------
# 2. Run SINDy
# -----------------------

library = ps.PolynomialLibrary(degree=1)

model = ps.SINDy(
    feature_library=library,
    optimizer=ps.STLSQ(threshold=1e-3)
)

model.fit(prices, u=flows, t=dt)

print("\nDiscovered equation:")
model.print()


# -----------------------
# 3. Simulate learned model
# -----------------------

examples = []

for k in range(3):

    p_true = prices[k].flatten()
    u = flows[k].flatten()

    p_pred = np.zeros(n_steps)
    p_pred[0] = p_true[0]

    for t in range(1,n_steps):

        dp = model.predict(p_pred[t-1].reshape(1,1),u[t].reshape(1,1))[0,0]
        p_pred[t] = p_pred[t-1] + dp*dt

    examples.append((p_true,p_pred))


# -----------------------
# 4. Plot comparison
# -----------------------

plt.figure(figsize=(8,5))

for i,(truth,pred) in enumerate(examples):

    color = f"C{i}"

    plt.plot(truth,color=color,label=f"truth {i+1}")
    plt.plot(pred,"--",color=color,label=f"sindy {i+1}")

plt.xlabel("time")
plt.ylabel("price")
plt.title("Ground truth vs SINDy prediction")

plt.legend()
plt.tight_layout()

plt.savefig("sindy_vs_truth.png",dpi=150)
plt.close()
