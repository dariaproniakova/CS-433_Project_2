# %% Package import

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# %% Import data
with open("dataset.pkl", "rb") as f:
    data = pickle.load(f)

n = len(data)
nbins = int(1 + 3.322 * np.log10(n))
# %% Define strain and plies FI values

eps = pd.DataFrame(
    [d["eps_global"] for d in data],
    columns=["11", "22", "33", "23", "13", "12"],
)

print("Strain values")
print(eps.head().round(4))

# %%
scaler = MinMaxScaler(feature_range=(-1, 1))
eps_scaled = eps.copy()
eps_scaled.iloc[:, :] = scaler.fit_transform(eps.values)

print("Strain values scaled")
print(eps_scaled.head().round(2))
# %%
plies = {}
angles = [0.0, 45.0, 90.0, -45.0]

for angle in angles:
    plies[angle] = pd.DataFrame(
        [d["plies"][angle] for d in data],
    )

print("Plies")
print(list(plies.keys()))
# %%
print("Failure index values for ply 0")
print(list(plies.values())[0].head().round(2))

# %% Faiure summary

stacked = pd.concat(plies, axis=1)
stacked.columns = [f"{angle}_{mode}" for angle, mode in stacked.columns]

max_val = stacked.max(axis=1)
max_col = stacked.idxmax(axis=1)
split = max_col.str.split("_", expand=True)
max_angle = split[0].astype(float)
short_mode = split[2]

values = stacked.to_numpy()
cols = stacked.columns.get_indexer(max_col)
FI = values[np.arange(len(values)), cols]

fail_threshold = 1.0
ffp = np.where(max_val >= fail_threshold, max_angle, np.nan)
mode = np.where(max_val >= fail_threshold, short_mode, "nf")

fail_summary = pd.DataFrame(
    {
        "ffp": np.where(np.isnan(ffp), "none", ffp),
        "mode": mode,
        "FI": FI,
    }
)

print("Failure summary")
print(fail_summary.sample(10, random_state=21))


# %% Failure Index Distributions per Ply

# fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
# axes = axes.ravel()
# n = len(data)
# # nbins = int(2 * n ** (1/3))
# nbins = int(1 + 3.322 * np.log10(n))
# # nbins = int(np.sqrt(n))
# for ax, angle in zip(axes, angles):
#     df = plies[angle]

#     ft = df["FI_ft"].values
#     fc = df["FI_fc"].values
#     mt = df["FI_mt"].values
#     mc = df["FI_mc"].values

#     # Filter out zeros to avoid clutter
#     ft = ft[ft != 0]
#     fc = fc[fc != 0]
#     mt = mt[mt != 0]
#     mc = mc[mc != 0]

#     all_vals = np.concatenate([ft, fc, mt, mc])
#     n = len(all_vals)
#     nbins = int(1 + 3.322 * np.log10(n))

#     ax.hist(
#         [ft, fc, mt, mc],
#         bins=nbins,
#         label=[
#             "ft",
#             "fc",
#             "mt",
#             "mc",
#         ],
#         color=["tab:blue", "tab:cyan", "tab:orange", "tab:red"],
#         alpha=0.7,
#         stacked=False,
#     )

#     ax.axvline(1, color="r", linewidth=0.8, linestyle="--")

#     ax.set_title(f"Ply {angle}°", fontsize=11)
#     ax.legend(fontsize=8)
#     ax.grid(True, linestyle=":", alpha=0.5)

# fig.suptitle("Failure Index Distributions per Ply", fontsize=14, weight="bold")
# fig.text(0.5, 0.04, "Failure index", ha="center", fontsize=12)
# fig.text(0.04, 0.5, "Count", va="center", rotation="vertical", fontsize=12)

# plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
# plt.show()

# %% Signed Failure Index Distributions per Ply


# def signed_index(ft, fc):
#     # Return +ft if tension, -fc if compression
#     return ft if ft > 0 else -fc


# plies_signed = {}

# for angle in angles:
#     fiber_signed = [
#         signed_index(ft, fc)
#         for ft, fc in zip(plies[angle]["FI_ft"], plies[angle]["FI_fc"])
#     ]
#     matrix_signed = [
#         signed_index(mt, mc)
#         for mt, mc in zip(plies[angle]["FI_mt"], plies[angle]["FI_mc"])
#     ]
#     plies_signed[angle] = np.column_stack([fiber_signed, matrix_signed])

# fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
# axes = axes.ravel()

# for ax, angle in zip(axes, angles):
#     fiber_vals = plies_signed[angle][:, 0]
#     matrix_vals = plies_signed[angle][:, 1]

#     ax.hist(
#         [fiber_vals[fiber_vals != 0], matrix_vals[matrix_vals != 0]],
#         # [fiber_vals, matrix_vals],
#         bins=2 * nbins,
#         label=["Fiber", "Matrix"],
#         color=["tab:blue", "tab:orange"],
#         alpha=0.7,
#     )

#     ax.axvline(0, color="k", linewidth=0.8)
#     ax.axvline(1, color="r", linewidth=0.8, linestyle="--")
#     ax.axvline(-1, color="r", linewidth=0.8, linestyle="--")
#     ax.set_title(f"Ply {angle}°", fontsize=11)
#     ax.legend()
#     ax.grid(True, linestyle=":", alpha=0.5)

# fig.suptitle("Signed Failure Index Distributions per Ply", fontsize=14, weight="bold")
# fig.text(0.5, 0.04, "Signed failure index", ha="center", fontsize=12)
# fig.text(0.04, 0.5, "Count", va="center", rotation="vertical", fontsize=12)
# # plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
# plt.tight_layout
# plt.show()
# %% Failure Index Distributions per Mode
# modes = ["FI_ft", "FI_fc", "FI_mt", "FI_mc"]
# labels = [
#     "Fiber tension: ft",
#     "Fiber compression: fc",
#     "Matrix tension: mt",
#     "Matrix compression: mc",
# ]
# angles = list(plies.keys())
# colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# # --- Prepare figure ---
# fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
# axes = axes.ravel()

# # --- Loop over failure modes ---
# for ax, mode, label in zip(axes, modes, labels):
#     # Collect all plies for this mode
#     data_per_ply = []
#     for angle in angles:
#         vals = plies[angle][mode].values
#         vals = vals[vals != 0]  # remove zeros for clarity
#         data_per_ply.append(vals)

#     # Determine bins based on total data size
#     n = sum(len(v) for v in data_per_ply)
#     nbins = int(1 + 3.322 * np.log10(n))  # Sturges' rule

#     # Plot overlaid histograms for each ply
#     ax.hist(
#         data_per_ply,
#         bins=nbins,
#         label=[f"{angle}°" for angle in angles],
#         color=colors[: len(angles)],
#         alpha=0.6,
#         stacked=False,
#     )
#     ax.axvline(1, color="r", linewidth=0.8, linestyle="--")
#     ax.set_title(label, fontsize=11)
#     ax.legend(fontsize=8)
#     ax.grid(True, linestyle=":", alpha=0.5)

# fig.suptitle(
#     "Failure Index Distributions per Mode (across plies)", fontsize=14, weight="bold"
# )
# fig.text(0.5, 0.04, "Failure index", ha="center", fontsize=12)
# fig.text(0.04, 0.5, "Count", va="center", rotation="vertical", fontsize=12)

# plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
# plt.show()

# %% Distribution of Fmax by First Failing Ply
# Ensure consistent typing
fail_summary["ffp"] = (
    fail_summary["ffp"].replace("none", np.nan).astype(float).fillna("none")
)

order = ["none", 0.0, 45.0, 90.0, -45.0]
palette = {
    "none": "#bdbdbd",
    -45.0: "#9ecae1",
    0.0: "#6a51a3",
    45.0: "#3182bd",
    90.0: "#9e9ac8",
}

ply_counts = fail_summary["ffp"].value_counts()
relative_widths = (ply_counts / ply_counts.max()).reindex(order).fillna(0)

fig, ax = plt.subplots(figsize=(8, 5))
for i, ply in enumerate(order):
    subset = fail_summary[fail_summary["ffp"] == ply]
    if not subset.empty:
        sns.violinplot(
            data=subset,
            x=[i] * len(subset),
            y="FI",
            inner="quartile",
            cut=0,
            bw_adjust=0.8,
            width=0.8 * relative_widths[ply],
            color=palette[ply],
            ax=ax,
        )

ax.set(
    xticks=range(len(order)),
    xticklabels=[str(p) for p in order],
    title="Distribution of Fmax by First Failing Ply",
    xlabel="First Failing Ply [°]",
    ylabel="Failure Index (Fmax)",
)
ax.axhline(
    1, color="black", linestyle="--", linewidth=1.5, label="Failure threshold (FI = 1)"
)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(loc="upper right")

for i, ply in enumerate(order):
    ax.text(
        i,
        0.032,
        f"n={ply_counts.get(ply, 0)}",
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.get_xaxis_transform(),
    )

plt.tight_layout()
plt.show()


# %% Distribution of Fmax by Failure Mode

order = ["nf", "ft", "fc", "mt", "mc"]
palette = {
    "nf": "#bdbdbd",
    "ft": "#6a51a3",
    "fc": "#9e9ac8",
    "mt": "#3182bd",
    "mc": "#9ecae1",
}

mode_counts = fail_summary["mode"].value_counts()
relative_widths = (mode_counts / mode_counts.max()).reindex(order).fillna(0.0)

fig, ax = plt.subplots(figsize=(8, 5))

for i, mode in enumerate(order):
    subset = fail_summary[fail_summary["mode"] == mode]
    if subset.empty:
        continue
    sns.violinplot(
        data=subset,
        x=[i] * len(subset),
        y="FI",
        inner="quartile",
        cut=0,
        bw_adjust=0.8,
        width=0.8 * relative_widths[mode],
        color=palette[mode],
        ax=ax,
    )

ax.set_xticks(range(len(order)))
ax.set_xticklabels(order)
ax.axhline(
    1.0,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Failure threshold (FI = 1)",
)
ax.set_title("Distribution of Fmax by Failure Mode")
ax.set_xlabel("Failure Mode")
ax.set_ylabel("Failure Index (Fmax)")
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(loc="upper right")

for i, mode in enumerate(order):
    count = mode_counts.get(mode, 0)
    ax.text(
        i,
        0.032,
        f"n={count}",
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.get_xaxis_transform(),
    )

plt.tight_layout()
plt.show()

# %% Maximum failure index distribution

plt.figure(figsize=(8, 5))
sns.histplot(data=fail_summary["FI"], bins=nbins, kde=True, color="steelblue")
plt.title("Distribution of Maximum Failure Index (Fmax)")
plt.xlabel("Failure Index (Fmax)")
plt.ylabel("Count")
plt.axvline(
    1.0,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Failure threshold (FI = 1)",
)
plt.tight_layout()
plt.show()


# %% Pairwise Plot: FI (lower), Mode (upper), KDE by Mode (diagonal)

clip_value = 2
scatter_FI_heatmap = "coolwarm"
max_points = 3000  # adjust for density

df = eps.copy()
# strain_cols = ["1", "2", "3", "23", "13", "12"]
strain_cols = [rf"$\varepsilon_{i}$" for i in ["1", "2", "3"]] + [
    rf"$\gamma_{{{j}}}$" for j in ["23", "13", "12"]
]

# Example nonlinear relationship for FI
df["FI"] = fail_summary["FI"].copy()
df["FI"] = np.clip(df["FI"].to_numpy(), 0, clip_value)
df["mode"] = fail_summary["mode"]

if len(df) > max_points:
    df_plot = df.sample(max_points, random_state=42)
else:
    df_plot = df.copy()

df_plot.columns = strain_cols + list(df.columns[6:])

df_plot[strain_cols] = scaler.fit_transform(df_plot[strain_cols])

modes = ["nf", "ft", "fc", "mt", "mc"]
sns.set_style("whitegrid")
# sns.set_style("darkgrid", {"axes.facecolor": "#000000"})

# ==============================================================
# Custom plotting functions
# ==============================================================


def scatter_fi(x, y, **kwargs):
    """Lower triangle: scatter colored by continuous FI."""
    plt.scatter(
        x,
        y,
        c=df_plot.loc[x.index, "FI"],
        cmap=scatter_FI_heatmap,
        s=10,
        alpha=1,
        linewidths=0,
    )


def scatter_mode(x, y, **kwargs):
    """Upper triangle: scatter colored by categorical mode."""
    sns.scatterplot(
        x=x,
        y=y,
        hue=df_plot.loc[x.index, "mode"],
        palette="tab10",
        s=10,
        alpha=0.6,
        linewidth=0,
        legend=False,
    )


def kde_mode(x, **kwargs):
    """Diagonal: KDE per mode, smooth & semi-transparent."""
    sns.kdeplot(
        data=df_plot,
        x=x.name,
        hue="mode",
        palette="tab10",
        common_norm=False,
        fill=True,
        alpha=0.1,  # matches scatter transparency
        lw=1,  # no hard line
        legend=False,
    )


# ==============================================================
# PairGrid setup
# ==============================================================

g = sns.PairGrid(df_plot, vars=strain_cols, diag_sharey=False)
g.map_lower(scatter_fi)
g.map_upper(scatter_mode)
g.map_diag(kde_mode)

# Shared FI colorbar
norm = plt.Normalize(vmin=df_plot["FI"].min(), vmax=df_plot["FI"].max())
sm = plt.cm.ScalarMappable(cmap=scatter_FI_heatmap, norm=norm)
sm.set_array([])
cax = g.fig.add_axes([0.92, 0.2, 0.02, 0.6])
g.fig.colorbar(sm, cax=cax, label="Failure Index (FI)")

import matplotlib.patches as mpatches

# Rebuild global legend manually from palette and mode list
palette = sns.color_palette("tab10", n_colors=len(modes))
legend_patches = [
    mpatches.Patch(color=palette[i], label=modes[i]) for i in range(len(modes))
]

g.fig.legend(
    handles=legend_patches,
    title="Failure Mode",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.95),
)


# Layout and title
g.fig.suptitle(
    f"Pairwise Scatter Plot: FI (lower), Mode (upper), KDE by Mode (diagonal) - n={max_points}",
    fontsize=14,
    weight="bold",
)
g.fig.subplots_adjust(top=0.95, right=0.9)

# Force real data limits on every subplot
for i, var_y in enumerate(strain_cols):
    for j, var_x in enumerate(strain_cols):
        ax = g.axes[i, j]
        if ax is None:
            continue
        if i != j:  # scatter plots
            ax.set_xlim(df_plot[var_x].min(), df_plot[var_x].max())
            ax.set_ylim(df_plot[var_y].min(), df_plot[var_y].max())
        else:  # diagonal plots (KDE)
            ax.set_xlim(df_plot[var_x].min(), df_plot[var_x].max())


plt.show()
# %% Others
# angles = [0.0, 45.0, 90.0, -45.0]
# np.set_printoptions(precision=6, suppress=True)
# k = 10
# print(f"Global strain sample #{k} (Voigt [xx, yy, zz, yz, xz, xy])")
# print("eps_global:", data[k]["eps_global"])

# for th in angles:
#     d = data[k]["plies"][th]
#     print(f"Ply {th:+.0f}°:")
#     # print(data[k]["plies"][th]["criteria"])
#     print(
#         "FI_ft, FI_fc, FI_mt, FI_mc:",
#         f"{d["FI_ft"]:.3f}",
#         f"{d["FI_fc"]:.3f}",
#         f"{d["FI_mt"]:.3f}",
#         f"{d["FI_mc"]:.3f}",
#     )
# Distribution of First Failing Ply by Mode
# plt.figure(figsize=(8, 5))
# sns.countplot(
#     data=fail_summary[fail_summary["mode"] != "nf"],
#     x="ffp",
#     hue="mode",
# )
# plt.title("Distribution of First Failing Ply by Mode")
# plt.xlabel("Ply angle")
# plt.ylabel("Count")
# plt.legend(title="Failure mode")
# plt.tight_layout()
# plt.show()

'''
df = eps.copy()

# Example nonlinear relationship for FI
df["FI"] = fail_summary["FI"].copy()
df["FI"] = np.clip(df["FI"].to_numpy(), 0, 2)

max_points = 3000  # adjust for density
if len(df) > max_points:
    df_plot = df.sample(max_points, random_state=42)
else:
    df_plot = df.copy()

# --- PairGrid approach (to allow continuous color mapping) ---
sns.set_style("whitegrid")
strain_cols = ["11", "22", "33", "23", "13", "12"]


def colored_scatter(x, y, **kwargs):
    """Scatter colored by FI (passed via global df_plot)."""
    plt.scatter(
        x,
        y,
        c=df_plot.loc[x.index, "FI"],
        cmap="viridis",
        s=10,
        alpha=0.6,
        linewidths=0,
    )


# --- Build PairGrid manually ---
g = sns.PairGrid(df_plot, vars=strain_cols, corner=True, diag_sharey=False)

# Map custom scatter function (colors by FI)
g.map_lower(colored_scatter)

# Diagonal histograms
g.map_diag(sns.histplot, bins=40, color="gray")

# Add a shared colorbar for FI
norm = plt.Normalize(vmin=df_plot["FI"].min(), vmax=df_plot["FI"].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

cax = g.fig.add_axes([0.92, 0.2, 0.02, 0.6])  # position of colorbar
g.fig.colorbar(sm, cax=cax, label="Failure Index (FI)")

# Titles and layout
g.fig.suptitle(
    "Pairwise Scatter Plot of Strain Components Colored by FI",
    fontsize=14,
    weight="bold",
)
g.fig.subplots_adjust(top=0.95, right=0.9)
plt.show()
'''
"""
df2 = eps.copy()
df2["mode"] = fail_summary["mode"]

# --- Downsample to keep the plot readable ---
max_points = 3000  # adjust for density
if len(df2) > max_points:
    df_plot = df2.sample(max_points, random_state=42)
else:
    df_plot = df2.copy()

# --- Sort by mode for better color grouping ---
df_plot = df_plot.sort_values(by="mode")

# --- Seaborn pairplot ---
sns.set_style("whitegrid")
g = sns.pairplot(
    df_plot,
    vars=["11", "22", "33", "23", "13", "12"],
    hue="mode",
    palette="tab10",
    plot_kws=dict(s=10, alpha=0.6, linewidth=0),
    corner=True,
)

g.fig.suptitle(
    "Downsampled Pairwise Scatter Plot by Failure Mode", fontsize=14, weight="bold"
)
g.fig.subplots_adjust(top=0.95)  # space for title
plt.show()
"""
