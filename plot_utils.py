# plot_utils.py
import matplotlib.pyplot as plt

def style_streamlit_plot(fig, ax):
    # --- ADD FRAME (spines) ---
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("#2C3E50")

    # --- Background ---
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)

    # --- BOLD titles and labels ---
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")

    # --- Bold tick labels ---
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    return fig