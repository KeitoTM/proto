import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("C:\\Users\\keito\\vscode\\k\\data\\dis.csv",encoding ="shift-jis",header = 1)
x = np.array(df["cx"])
peak_value = np.array(df["peak value"])
r = 0.0427
dt = np.float32(1/60)
x -= x[0]
#x*= r
t = np.arange(0,dt*len(x),len(x))
plt.plot(x)
plt.grid()
plt.show()
plt.plot(peak_value)
plt.ylim(0,1)
plt.grid()
plt.show()