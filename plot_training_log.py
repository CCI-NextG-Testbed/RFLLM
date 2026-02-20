import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv")

plt.figure()

for attn, g in df.groupby("attn_type"):
    g = g.sort_values("epoch")
    plt.plot(g["epoch"], g["mean_loss"], marker="o", label=attn)

plt.xlabel("Epoch")
plt.ylabel("Mean Training Loss")
plt.title("Loss vs Epoch (Attention Comparison)")
plt.legend()
plt.grid(True)
plt.show()
