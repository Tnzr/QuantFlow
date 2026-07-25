"""MAE-by-step evolution chart: one line per epoch, showing forecast accuracy
improving over training. The x-axis is forecast distance (steps), the y-axis
is MAE %. Early epochs show higher error (red), later epochs lower (green)."""
import io
import numpy as np


def build_mae_evolution_figure(mae_history: dict, n_steps: int) -> object:
    """Build a matplotlib figure showing MAE-by-step evolution across epochs.

    Args:
        mae_history: dict {epoch_num: [mae_step0, mae_step1, ..., mae_step_N]}
        n_steps: total number of forecast steps

    Returns:
        wandb.Image or None
    """
    if not mae_history:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        epochs = sorted(mae_history.keys())
        cmap = plt.cm.RdYlGn_r  # red→yellow→green (red=early, green=late)
        colors = cmap(np.linspace(0, 1, len(epochs)))

        steps = np.arange(n_steps)

        for i, ep in enumerate(epochs):
            mae_vec = mae_history[ep]
            if len(mae_vec) < n_steps:
                mae_vec = list(mae_vec) + [mae_vec[-1]] * (n_steps - len(mae_vec))
            mae_pct = np.array(mae_vec[:n_steps]) * 100  # to %
            alpha = 0.4 + 0.6 * (i / max(1, len(epochs) - 1))
            ax.plot(steps, mae_pct, color=colors[i], linewidth=1.2, alpha=alpha,
                    label=f"Epoch {ep}" if i % max(1, len(epochs) // 8) == 0 or i == len(epochs) - 1 else "")

        ax.set_xlabel("Forecast step (t)")
        ax.set_ylabel("MAE %")
        ax.set_title(f"Forecast MAE by Step — {len(epochs)} epochs")
        ax.legend(fontsize=6, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.15, linestyle="--")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=100)
        plt.close(fig)
        buf.seek(0)

        import wandb
        from PIL import Image
        return wandb.Image(Image.open(buf), caption="MAE by forecast step — evolution across epochs")
    except Exception as exc:
        print(f"MAE evolution viz failed: {exc}")
        return None
