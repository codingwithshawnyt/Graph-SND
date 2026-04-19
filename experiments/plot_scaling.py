"""Plot SND_full and GraphSND_p vs iter with speedup."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/scaling/n100_overnight_snd_log.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(df["iter"], df["SND_full"], "o-",
         label=r"$\mathrm{SND}(\mathbf{D})$", color="#1f77b4", linewidth=2)
ax1.plot(df["iter"], df["GraphSND_p"], "s--",
         label=r"$\widehat{\mathrm{SND}}(G_{0.1})$",
         color="#ff7f0e", linewidth=2, alpha=0.8)
ax1.set_xlabel("PPO iteration")
ax1.set_ylabel("diversity")
ax1.set_title(r"$\mathrm{SND}$ and $\widehat{\mathrm{SND}}(G_{0.1})$ during training, $n{=}100$")
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)

ax2.plot(df["iter"], df["speedup"], "o-", color="#2ca02c", linewidth=2)
ax2.axhline(10, color="gray", linestyle=":", alpha=0.7,
            label=r"predicted $1/p = 10\times$")
ax2.set_xlabel("PPO iteration")
ax2.set_ylabel(r"speedup $T_{\mathrm{full}} / T_{\mathrm{sample}}$")
ax2.set_title(r"wall-clock speedup during training")
ax2.legend(loc="lower right")
ax2.grid(alpha=0.3)
ax2.set_ylim(bottom=0)

fig.tight_layout()
out = Path("results/scaling/scaling_n100.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")