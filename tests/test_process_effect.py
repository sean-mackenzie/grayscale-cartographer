import pandas as pd
import matplotlib.pyplot as plt

from graycart.utils import process


d_fn = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer11/results/w11_merged_process_profiles.xlsx'
dff = pd.read_excel(d_fn)

steps = [4, 5]
fid = 0

for step in steps:
    df1 = dff[(dff['step'] == step - 1) & (dff['fid'] == fid)]
    df2 = dff[(dff['step'] == step) & (dff['fid'] == fid)]

    x_new, y1, y2 = process.uniformize_x_dataframes(dfs=[df1, df2], xy_cols=['r', 'z'], num_points=None, sampling_rate=10)
    dy = y1 - y2

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    ax1.plot(df1.r, df1.z, '.', ms=0.5, alpha=0.5)
    ax1.plot(df2.r, df2.z, '.', ms=0.5, alpha=0.5)
    ax1.plot(x_new, y1, linewidth=0.5, color='b', label=step-1)
    ax1.plot(x_new, y2, linewidth=0.5, color='darkgreen', label=step)
    ax1.legend(title='Step')

    ax2.plot(x_new, dy, label=r'$\Delta_{ij}$')
    ax2.grid(alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()