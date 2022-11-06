import matplotlib.pyplot as plt
import pandas as pd
from sheina.consts import *

holter_labels = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/Holter_label_new2.xlsx', engine='openpyxl')
shany_ids = np.load('/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/reports/control_group.npy')
keywords = pd.read_excel(rbdb_doc_path + 'reports/RBAF_reports.xlsx', engine='openpyxl')

holter_labels['Unnamed: 0'] = holter_labels['Unnamed: 0'].astype(str)
keywords['holter_id'] = keywords['holter_id'].astype(str)

for i, holter in enumerate(shany_ids):
    j = int(holter_labels['Unnamed: 0'][holter_labels['Unnamed: 0'] == str(holter)].index.values)
    print(holter_labels['AFIB'][j])

# for AFIB:

fig, ax = plt.subplots()
data = np.asarray(holter_labels['AFIB'])
N, bins, patches = ax.hist(data, edgecolor='white', linewidth=1, bins=100)
for i in range(90, 100):
    patches[i].set_facecolor('r')
plt.axvline(80, color='k', linestyle='dashed', linewidth=1)
ax.set_title('AFIB')
ax.set_xlabel('grade')
ax.set_ylabel('# holters')
plt.show()

# for VT
fig, ax = plt.subplots()
data = np.asarray(holter_labels['VT'])
data[data < 0] = 0
N, bins, patches = ax.hist(data, edgecolor='white', linewidth=1, bins=100)
patches[-1].set_facecolor('r')
plt.axvline(98, color='k', linestyle='dashed', linewidth=1)
ax.set_title('VT')
ax.set_xlabel('grade')
ax.set_ylabel('# holters')
plt.show()

afib_holters = holter_labels[holter_labels['AFIB'] >= 80]
vt_holters = holter_labels[holter_labels['VT'] >= 100]

may_afib_holters = holter_labels[holter_labels['AFIB'] == 75]

MAFH = pd.DataFrame(columns=['AF score', 'text a', 'text b'],
                    index=afib_holters['Unnamed: 0'])  # high scored VT holters

for i, holter in enumerate(afib_holters['Unnamed: 0']):
    j = int(keywords['holter_id'][keywords['holter_id'] == holter].index.values)
    MAFH['text a'][i] = keywords['טקסט מתוך סיכום'][j]
    MAFH['text b'][i] = keywords['טקסט מתוך תוצאות'][j]

    jvt = int(afib_holters['Unnamed: 0'][afib_holters['Unnamed: 0'] == holter].index.values)
    MAFH['AF score'][i] = afib_holters['AFIB'][jvt]
MAFH.to_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/keyword_ans/AFH.xlsx')
a = 5
