import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Setup: create folder for saving plots
# ===============================
SAVE_DIR = "exploratory_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# 1️⃣ Distribution graphs (histogram/bar graph)
# ===============================
def plotPerColumnDistribution(df, file_name, nGraphShown, nGraphPerRow):
    """
    Plots histograms for numeric columns and bar charts for categorical columns.
    Shows up to nGraphShown columns, with nGraphPerRow columns per row.
    Saves the plot as PNG in exploratory_analysis/.
    """
    nunique = df.nunique()
    df = df[[col for col in df if 1 < nunique[col] < 50]]
    nRow, nCol = df.shape
    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow

    plt.figure(
        num=None,
        figsize=(6 * nGraphPerRow, 8 * nGraphRow),
        dpi=80,
        facecolor="w",
        edgecolor="k",
    )

    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if np.issubdtype(columnDf.dtype, np.number):
            columnDf.hist()
        else:
            columnDf.value_counts().plot.bar()
        plt.ylabel("counts")
        plt.xticks(rotation=90)
        plt.title(f"{columnNames[i]} (column {i})")

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    save_path = os.path.join(SAVE_DIR, f"{file_name}_distribution.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved distribution plot: {save_path}")
    plt.show()
    plt.close()


# ===============================
# 2️⃣ Correlation matrix
# ===============================
def plotCorrelationMatrix(df, file_name, graphWidth):
    """
    Plots correlation matrix for numeric columns only.
    Saves the plot as PNG in exploratory_analysis/.
    """
    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis="columns")
    df = df[[col for col in df if df[col].nunique() > 1]]

    if df.shape[1] < 2:
        print(f"No correlation plots shown: {df.shape[1]} numeric columns available")
        return

    corr = df.corr()
    plt.figure(
        num=None,
        figsize=(graphWidth, graphWidth),
        dpi=80,
        facecolor="w",
        edgecolor="k",
    )
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f"Correlation Matrix for {file_name}", fontsize=15)
    save_path = os.path.join(SAVE_DIR, f"{file_name}_correlation.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved correlation plot: {save_path}")
    plt.show()
    plt.close()


# ===============================
# 3️⃣ Scatter and density plots
# ===============================
def plotScatterMatrix(df, file_name, plotSize=12, textSize=10):
    """
    Scatter matrix with max 10 numeric columns.
    Saves plot safely with bbox_inches='tight'.
    """
    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis="columns")
    df = df[[col for col in df if df[col].nunique() > 1]]

    columnNames = list(df)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
        df = df[columnNames]

    # Reduce figure size if too large
    plotSize = min(plotSize, 15)

    ax = pd.plotting.scatter_matrix(
        df, alpha=0.75, figsize=[plotSize, plotSize], diagonal="kde"
    )
    corrs = df.corr().values

    for i, j in zip(*np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate(
            f"Corr. coef = {corrs[i, j]:.3f}",
            (0.8, 0.2),
            xycoords="axes fraction",
            ha="center",
            va="center",
            size=textSize,
        )

    plt.suptitle(f"Scatter and Density Plot for {file_name}")
    save_path = os.path.join(SAVE_DIR, f"{file_name}_scatter.png")
    plt.savefig(save_path, bbox_inches="tight")  # important for large figures
    print(f"Saved scatter matrix plot: {save_path}")
    plt.close()


# ===============================
# Load and analyze CSV files
# ===============================
def analyze_csv(
    file_path,
    nrows=None,
    nGraphShown=10,
    nGraphPerRow=5,
    corrSize=10,
    scatterSize=12,
    scatterText=10,
):
    """
    Reads CSV and generates distribution, correlation, and scatter plots.
    """
    df = pd.read_csv(file_path, nrows=nrows)
    file_name = os.path.basename(file_path).replace(".csv", "")
    nRow, nCol = df.shape
    print(f"\nFile: {file_path} | Rows: {nRow}, Columns: {nCol}")
    df.head(5)

    plotPerColumnDistribution(df, file_name, nGraphShown, nGraphPerRow)
    plotCorrelationMatrix(df, file_name, corrSize)
    plotScatterMatrix(df, file_name, scatterSize, scatterText)
    return df


# ===============================
# Run analysis on all 3 files
# ===============================
df1 = analyze_csv(
    "data/celeba/list_attr_celeba.csv",
    nrows=1000,
    nGraphShown=10,
    nGraphPerRow=5,
    corrSize=10,
    scatterSize=20,
    scatterText=10,
)
df2 = analyze_csv(
    "data/celeba/list_bbox_celeba.csv",
    nrows=1000,
    nGraphShown=10,
    nGraphPerRow=5,
    corrSize=8,
    scatterSize=12,
    scatterText=10,
)
