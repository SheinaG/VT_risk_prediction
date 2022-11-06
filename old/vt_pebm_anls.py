import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sheina.consts import INTS, WAVES

plt.rcParams.update({'font.size': 15})

with open('/MLdata/AIMLab/Sheina/databases/VTdb/VTp/pebm_vt.npy', 'rb') as handle:
    bm_vt = pickle.load(handle)
bm_vt_pd = pd.DataFrame.from_dict(bm_vt)
bm_vt_pd = bm_vt_pd.transpose()

with open('/MLdata/AIMLab/Sheina/databases/VTdb/VTn/pebm_no_vt.npy', 'rb') as handle:
    bm_no_vt = pickle.load(handle)
bm_no_vt_pd = pd.DataFrame.from_dict(bm_no_vt)
bm_no_vt_pd = bm_no_vt_pd.transpose()

plt.style.use('bmh')
color = ['#003049']
edg_color = ["#0D232E"]

for int in INTS:
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(25, 10))
    for lead in [0, 1, 2]:

        data1 = bm_vt_pd['median_' + int + '_lead_' + str(lead)]
        ax1 = axes[lead, 0]
        violin1 = ax1.violinplot(data1, showmeans=False, showmedians=True, showextrema=True)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp = violin1[partname]
            vp.set_edgecolor(edg_color)
        violin1['bodies'][0].set_facecolor(color)
        violin1['bodies'][0].set_alpha(0.8)

        ax1.set_title('median of ' + int + ' lead #' + str(lead) + ',$VT$')
        ax1.set_ylabel('Time (ms)')

        data2 = bm_no_vt_pd['median_' + int + '_lead_' + str(lead)]
        ax2 = axes[lead, 1]
        violin2 = ax2.violinplot(data2, showmeans=False, showmedians=True, showextrema=True)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp = violin2[partname]
            vp.set_edgecolor(edg_color)
        violin2['bodies'][0].set_facecolor(color)
        violin2['bodies'][0].set_alpha(0.8)

        ax2.set_title('median of ' + int + ' lead #' + str(lead) + ',$NO_VT$')
        ax2.set_ylabel('Time (ms)')
    fig.tight_layout()

    plt.savefig('/MLdata/AIMLab/Sheina/databases/VTdb/figures/violin_plot_' + int + '.png', dpi=400, transparent=True)
    plt.show()

for wave in WAVES:
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(25, 10))
    for lead in [0, 1, 2]:

        data1 = bm_vt_pd['median_' + wave + '_lead_' + str(lead)]
        ax1 = axes[lead, 0]
        violin1 = ax1.violinplot(data1, showmeans=False, showmedians=True, showextrema=True)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp = violin1[partname]
            vp.set_edgecolor(edg_color)
        violin1['bodies'][0].set_facecolor(color)
        violin1['bodies'][0].set_alpha(0.8)

        ax1.set_title('median of ' + wave + ' lead #' + str(lead) + ',$VT$')
        ax1.set_ylabel('Time (ms)')

        data2 = bm_no_vt_pd['median_' + wave + '_lead_' + str(lead)]
        ax2 = axes[lead, 1]
        violin2 = ax2.violinplot(data2, showmeans=False, showmedians=True, showextrema=True)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp = violin2[partname]
            vp.set_edgecolor(edg_color)
        violin2['bodies'][0].set_facecolor(color)
        violin2['bodies'][0].set_alpha(0.8)

        ax2.set_title('median of ' + wave + ' lead #' + str(lead) + ',$NO_VT$')
        ax2.set_ylabel('Time (ms)')
    fig.tight_layout()

    plt.savefig('/MLdata/AIMLab/Sheina/databases/VTdb/figures/violin_plot_' + wave + '.png', dpi=400, transparent=True)
    plt.show()
a = 5
