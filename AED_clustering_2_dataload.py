import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys 
import tqdm
from utils_AED import *
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

raw_data_dir = 'Data.xlsx'
raw_clinical = pd.read_excel(raw_data_dir, sheet_name='clinical')
raw_lab_EEG_MRI = pd.read_excel(raw_data_dir, sheet_name='lab_EEG_MRI')
raw_AED_outcome = pd.read_excel(raw_data_dir, sheet_name='outcome')

print('clinical:',len(raw_clinical))
print('lab_EEG_MRI:',len(raw_lab_EEG_MRI))
print('AED_outcome:',len(raw_AED_outcome))

cli_lab = pd.concat([raw_clinical,raw_lab_EEG_MRI.iloc[:,1:]],axis=1)
AED = []
for ii in range(0,26):
    AED.append(pd.concat([raw_AED_outcome.iloc[:]['final outcome'], raw_AED_outcome.iloc[:,ii*2+2:ii*2+4],cli_lab.iloc[:,1:]],axis=1))

num_patient = pd.DataFrame(np.array(list(np.array(list(range(len(raw_AED_outcome))))+1)),columns={'patient No.'})
num_round = []
for ii in range(26):
    temp = np.ones(len(raw_AED_outcome), dtype=int)
    num_round.append(pd.DataFrame(temp+ii,columns={str(ii+1)+'round'}))
for ii in range(len(AED)):
    AED[ii]=pd.concat([num_patient, num_round[ii], AED[ii]],axis=1)
AED_np = []
for ii in range(len(AED)):
    AED_np.append(AED[ii].values)
AED_np_all = np.vstack(AED_np)
AED_bool = []
for ii in range(len(AED_np_all)):
    AED_bool.append(isNaN(AED_np_all[ii,3]))
AED_bool_np = np.hstack(AED_bool)
AED_sel = AED_np_all[AED_bool_np,:]
col_AED = AED_sel[:,3]
for ii in range(len(col_AED)):
    col_AED[ii] = col_AED[ii].replace(' ','')
    col_AED[ii] = col_AED[ii].replace('.', ',')
    col_AED[ii] = col_AED[ii].replace(',,', ',')
for ii in range(len(col_AED)):
    col_AED[ii] = col_AED[ii].replace('null(doctor)', 'NULL')
    col_AED[ii] = col_AED[ii].replace('null(self)', 'NULL')
    col_AED[ii] = col_AED[ii].replace('null(unclear)', 'NULL') 
temp_all = []
for ii in range(len(col_AED)):
    temp_all.append(col_AED[ii].split(','))
flat_list = []
for sublist in temp_all:
    for item in sublist:
        if item == '':
            print(sublist)
        flat_list.append(item)
np.where(np.array(flat_list) =='')
AED_list = np.array(sorted((list(set(flat_list)))))
AED_feat_all = []
for ii in range(len(AED_sel)):
    temp_AED = AED_sel[ii,3].split(',')
    AED_feat = np.zeros(len(AED_list),dtype=int)
    for jj in range(len(temp_AED)):
        AED_idx = np.where(temp_AED[jj] == AED_list)[0]
        AED_feat[AED_idx] = int(1)
    AED_feat_all.append(AED_feat)
AED_feat_all_np = np.vstack(AED_feat_all)
col_name1 = ['patient', 'round', 'final outcome']
col_name2 = ['point outcome']
col_name3 = list(AED_list)
col_name4 = AED[0].columns[5:]
pd1 = pd.DataFrame(AED_sel[:,0:3],columns=col_name1) 
pd2 = pd.DataFrame(AED_sel[:,4],columns=col_name2)
pd3 = pd.DataFrame(AED_feat_all_np,columns=col_name3)
pd4 = pd.DataFrame(AED_sel[:,5:],columns=col_name4)
AEDdata=pd.concat([pd1,pd2,pd3,pd4],axis=1)
AEDdata = AEDdata.reset_index()
del AEDdata['index']
categs_all = list(AEDdata.columns)
dtype_data = AEDdata.dtypes
AEDdata['patient'] = AEDdata['patient'].astype(int)
AEDdata['round'] = AEDdata['round'].astype(int)
AEDdata['final outcome'] = AEDdata['final outcome'].astype(int)
AEDdata['point outcome'] = AEDdata['point outcome'].astype(int)
AEDdata['Onset age'] = AEDdata['Onset age'].astype(float)
AEDdata['No. of seizures before AED initiation'] = AEDdata['No. of seizures before AED initiation'].astype(float)
AEDdata['disease duration'] = AEDdata['disease duration'].astype(float)
AEDdata['타원에서 AED 복용기간'] = AEDdata['타원에서 AED 복용기간'].astype(int)
AEDdata['Seizure classification'] = AEDdata['Seizure classification'].astype(int)
AEDdata['Epilepsy classification'] = AEDdata['Epilepsy classification'].astype(int)
AEDdata['Etiology'] = AEDdata['Etiology'].astype(int)
AEDdata['Hx of febrile convulsion'] = AEDdata['Hx of febrile convulsion'].astype(int)
AEDdata['FHx of epilepsy'] = AEDdata['FHx of epilepsy'].astype(int)
AEDdata['Op'] = AEDdata['Op'].astype(int)
AEDdata.iloc[:,36:69] = AEDdata.iloc[:,36:69].astype(float)
AEDdata.iloc[:,69:] = AEDdata.iloc[:,69:].fillna(-1)
dtype_data = AEDdata.dtypes
adverseeffect_c = []
for ii in range(len(AEDdata)):
    if AEDdata['point outcome'][ii] == 3:
        adverseeffect_c.append(ii)
AEDdata = AEDdata.drop(adverseeffect_c)
AEDdata = AEDdata.reset_index(drop=True)