import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 200
age = rng.integers(18, 90, n)
bmi = np.round(rng.uniform(18.0, 45.0, n), 1)
glucose = np.round(rng.uniform(70.0, 220.0, n), 1)

# plausible risk logic: glucose/bmi/age push risk up
risk_prob = 1 / (1 + np.exp(-(0.04 * (glucose - 110) + 0.08 * (bmi - 27) + 0.02 * (age - 45))))
high_risk = (rng.random(n) < risk_prob).astype(int)

df = pd.DataFrame({"age": age, "bmi": bmi, "glucose": glucose, "high_risk": high_risk})
df.to_csv("diabetes_risk.csv", index=False)
print("diabetes_risk.csv created:", df.shape, "| high_risk rate:", round(df.high_risk.mean(), 3))
