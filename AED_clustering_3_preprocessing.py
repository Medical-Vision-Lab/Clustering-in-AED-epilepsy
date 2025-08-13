for ii in range(len(AEDdata)):
    if AEDdata.iloc[ii, 23] == 'M ':
        AEDdata.iloc[ii, 23] = 'M'
        print(ii)
AEDdata['Sex'].replace({'M':0,'F':1},inplace=True)
AEDdata['Sex'] = AEDdata['Sex'].astype(int)
como_all = []
for idx, como in enumerate(AEDdata['Comorbidities']):
    if como == 'N':
        como_all.append(0)
    else:
        como_all.append(1)
AEDdata['Comorbidities'] = como_all
AEDdata['Comorbidities'] = AEDdata['Comorbidities'].astype(int)
medi_all = []
for idx, medi in enumerate(AEDdata['Medication']):
    if medi == 'N':
        medi_all.append(0)
    else:
        medi_all.append(1)
AEDdata['Medication'] = medi_all
AEDdata['Medication'] = AEDdata['Medication'].astype(int)
AEDdata.dtypes
for idx, waveform in enumerate(AEDdata['Waveform1']):
    if waveform == 'N':
        AEDdata.iloc[idx, 70:78] = 'N'
# 1-2
for idx, waveform in enumerate(AEDdata['Slow4']):
    if waveform == 'N':
        AEDdata.iloc[idx, 79:84] = 'N'
# 1-3
for idx, waveform in enumerate(AEDdata['MRI_code1']):
    if waveform == 0:
        AEDdata.iloc[idx, 85:] = 0
# 1-4
llist = ['MRI_code2', 'MRI_code3', 'MRI_code4', 'MRI_code5']
for idx in range(len(AEDdata)):
    if AEDdata['MRI_code1'][idx] != 0 and AEDdata['MRI_code1'][idx] != -1:
        for catego in llist:
            if AEDdata[catego][idx] == -1:
                AEDdata[catego][idx] = 0
for idx, waveform in enumerate(AEDdata['Waveform1']):
    if waveform == -1:
        AEDdata.iloc[idx, 70:78]
    if waveform != -1 and AEDdata['Waveform2'][idx] == -1:
        AEDdata.iloc[idx, 73:78] = 'N'
    if waveform != 'Missing Value' and AEDdata['Waveform2'][idx] != -1 and AEDdata['Waveform3'][idx] == -1:
        AEDdata.iloc[idx, 76:78] = 'N'
for idx, waveform in enumerate(AEDdata['Slow4']):
    if AEDdata['Slow4'][idx] == -1:
        AEDdata.iloc[idx, 79:84] = 'N'
    if AEDdata['Slow4'][idx] != -1 and AEDdata['Slow5'][idx] == -1 :
        AEDdata.iloc[idx, 82:84] = 'N'
for liter in ['Laterality1','Laterality2','Laterality3','Laterality4','Laterality5']:
    for idx in range(len(AEDdata)):
        if AEDdata[liter][idx] == 'RL':
            AEDdata[liter][idx] = 'R'
        if AEDdata[liter][idx] == 'LR':
            AEDdata[liter][idx] = 'L'
for lobe in ['Lobes2']:
    for idx in range(len(AEDdata)):
        if AEDdata[lobe][idx] == 'p':
            AEDdata[lobe][idx] = 'P'
for liter in ['Laterality1','Laterality2','Laterality3','Laterality4','Laterality5']:
    for idx in range(len(AEDdata)):
        if AEDdata[liter][idx] == 'RL':
            AEDdata[liter][idx] = 'R'
        if AEDdata[liter][idx] == 'LR':
            AEDdata[liter][idx] = 'L'
for lobe in ['Lobes2']:
    for idx in range(len(AEDdata)):
        if AEDdata[lobe][idx] == 'p':
            AEDdata[lobe][idx] = 'P'
col_EEG_MRI = AEDdata.columns[69:]
types_all = {}
for col in col_EEG_MRI:
    types_each = list(set(AEDdata[col].values))
    types_all[col] = types_each
col_EEG = list(AEDdata.columns)[69:84]
type_EEG_col = {}
for col in col_EEG:
    type_EEG_col[col] = set(AEDdata[col])
waveform_categ = ['Waveform1','Waveform2','Waveform3']
lateral_categ = ['Laterality1','Laterality2','Laterality3']
waveform_col_name = ['Wave_RSW_G', 'Wave_SW_G', 'Wave_RDA_G', 'Wave_PD_G',
'Wave_RSW_NG', 'Wave_SW_NG', 'Wave_RDA_NG', 'Wave_PD_NG']
new_waveform = np.zeros((len(AEDdata),len(waveform_col_name)),dtype=int)
new_waveform = pd.DataFrame(new_waveform,columns=waveform_col_name)
for row in range(len(AEDdata)):
    for wave,lateral in zip(waveform_categ,lateral_categ):
        if AEDdata[wave][row] != 'N' and AEDdata[wave][row] != -1:
            if AEDdata[lateral][row] == 'G':
                if AEDdata[wave][row] == 'RDA':
                    new_waveform['Wave_RDA_G'][row] = 1
                elif AEDdata[wave][row] == 'RSW':
                    new_waveform['Wave_RSW_G'][row] = 1
                elif AEDdata[wave][row] == 'SW':
                    new_waveform['Wave_SW_G'][row] = 1
                else:
                    new_waveform['Wave_PD_G'][row] = 1
            else:
                if AEDdata[wave][row] == 'RDA':
                    new_waveform['Wave_RDA_NG'][row] = 1
                elif AEDdata[wave][row] == 'RSW':
                    new_waveform['Wave_RSW_NG'][row] = 1
                elif AEDdata[wave][row] == 'SW':
                    new_waveform['Wave_SW_NG'][row] = 1
                else:
                    new_waveform['Wave_PD_NG'][row] = 1
        else:
            if wave == 'Waveform1' and AEDdata[wave][row] == -1:
                new_waveform.iloc[row,:] = -1

slow_categ = ['Slow4','Slow5']
lateral_categ = ['Laterality4','Laterality5']
waveform_col_name = ['Slow_CS_G', 'Slow_IS_G', 'Slow_CS_NG', 'Slow_IS_NG']
new_slow = np.zeros((len(AEDdata),len(waveform_col_name)),dtype=int)
new_slow = pd.DataFrame(new_slow,columns=waveform_col_name)

for row in range(len(AEDdata)):
    for wave,lateral in zip(slow_categ,lateral_categ):
        if AEDdata[wave][row] != 'N' and AEDdata[wave][row] != -1:
            if AEDdata[lateral][row] == 'G':
                if AEDdata[wave][row] == 'CS':
                    new_slow['Slow_CS_G'][row] = 1
                else:
                    new_slow['Slow_IS_G'][row] = 1
            else:
                if AEDdata[wave][row] == 'CS':
                    new_slow['Slow_CS_NG'][row] = 1
                else:
                    new_slow['Slow_IS_NG'][row] = 1
        else:
            if wave == 'Slow4' and AEDdata[wave][row] == -1:
                new_slow.iloc[row,:] = -1
mri_categ = ['MRI_code1','MRI_code2','MRI_code3','MRI_code4','MRI_code5']
mri_col_name = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,99]
mri = np.zeros((len(AEDdata),len(mri_col_name)),dtype=int)
for row in range(len(mri)):
    mritypes = []
    for wave in mri_categ:
        mritypes.append(AEDdata[wave][row])
    if -1 in mritypes:
        assert len(list(set(mritypes))) == 1, print(row)
        mri[row, :] = -1
    else:
        if 1 in mritypes:
            mri[row, 0] = 1
        if 2 in mritypes:
            mri[row, 1] = 1
        if 3 in mritypes:
            mri[row, 2] = 1
        if 4 in mritypes:
            mri[row, 3] = 1
        if 5 in mritypes:
            mri[row, 4] = 1
        if 6 in mritypes:
            mri[row, 5] = 1
        if 7 in mritypes:
            mri[row, 6] = 1
        if 8 in mritypes:
            mri[row, 7] = 1
        if 9 in mritypes:
            mri[row, 8] = 1
        if 10 in mritypes:
            mri[row, 9] = 1
        if 11 in mritypes:
            mri[row, 10] = 1
        if 12 in mritypes:
            mri[row, 11] = 1
        if 13 in mritypes:
            mri[row, 12] = 1
        if 14 in mritypes:
            mri[row, 13] = 1
        if 15 in mritypes:
            mri[row, 14] = 1
        if 16 in mritypes:
            mri[row, 15] = 1
        if 99 in mritypes:
            mri[row, 16] = 1
pd_mri = pd.DataFrame(mri, columns=mri_col_name)
categs_eegmri = list(AEDdata.columns)[69:]
AEDdata = AEDdata.drop(categs_eegmri,axis='columns')
AEDdata = pd.concat([AEDdata,new_waveform, new_slow,pd_mri],axis=1)
AED_list = list(AEDdata.columns[4:23])
AED_num = []
for AED in AED_list:
    no_all = len(AEDdata[AED][AEDdata[AED] == 1])
    AED_num.append([no_all])
AED_num = np.array(AED_num)
AED_list_sel = list(np.array(AED_list)[AED_num[:,0] < 100])
sum(AED_num[AED_num<100])
for aed in AED_list_sel:
    AEDdata = AEDdata.loc[AEDdata[aed] == 0]
AEDdata = AEDdata.drop(AED_list_sel,axis='columns').reset_index(drop=True)
for op in range(AEDdata.shape[0]):
    if AEDdata['Op'][op] == 1:
        AEDdata['final outcome'][op] = 1
AEDdata = AEDdata.drop(['Op'],axis=1)
AEDdata.rename(columns = {'타원에서 AED 복용기간' : 'AED_duration_others'}, inplace = True)
patients = np.array(sorted(list(set(AEDdata['patient'].values))))
excet_pati = []
idx_fo = []
for patient in patients:
    p_idx = AEDdata['patient'] == patient
    p_ridx_max = AEDdata['round'][p_idx].idxmax()
    p_fo = AEDdata['final outcome'].astype(int)[p_ridx_max]
    p_po = AEDdata['point outcome'].astype(int)[p_ridx_max]
    if p_fo != p_po:
        print('Patient No.', patient)
        print('Last point outcome:', p_po)
        print('Final outcome:', p_fo)
        excet_pati.append(patient)
    idx_fo.append(p_ridx_max)
    
data_label = copy.deepcopy(AEDdata)
data_label = issue_yes_no(data_label)
categorical_columns = list(data_label.columns[0:16]) + list(data_label.columns[19:26]) + list(data_label.columns[60:])
numerical_columns = data_label.columns.difference(categorical_columns)
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_combined_data = imputer.fit_transform(data_label)
data_label.loc[:, :] = imputed_combined_data
data_label[categorical_columns] = data_label[categorical_columns].round()
data_label['Etiology'] = np.where(data_label['Etiology'] < 1, 1, data_label['Etiology'])
data_label['Etiology'] = np.where(data_label['Etiology'] > 6, 6, data_label['Etiology'])
epil_columns=['Epilepsy classification1', 'Epilepsy classification2', 'Epilepsy classification3']
etio_columns=['Etiology1','Etiology2', 'Etiology3','Etiology4', 'Etiology5', 'Etiology6']
column_ec = data_label['Epilepsy classification'].values.reshape(-1,1)
column_et = data_label['Etiology'].values.reshape(-1,1)
ohe_interest = ['Epilepsy classification', 'Etiology']
ohe = OneHotEncoder(sparse=False)
ohe.fit(column_ec)
df_ohe_ec = pd.DataFrame(ohe.transform(column_ec), columns=['Epilepsy classification1', 'Epilepsy classification2', 'Epilepsy classification3'])
ohe = OneHotEncoder(sparse=False)
ohe.fit(column_et)
df_ohe_et = pd.DataFrame(ohe.transform(column_et), columns=['Etiology1','Etiology2', 'Etiology3','Etiology4', 'Etiology5', 'Etiology6']) 
data_label = data_label.drop('Epilepsy classification', axis = 1)
data_label = data_label.drop('Etiology', axis = 1)
c_list_ml= list(data_label.columns)
data_label = pd.concat([data_label,df_ohe_ec], ignore_index=True, axis = 1)
data_label = pd.concat([data_label,df_ohe_et], ignore_index=True, axis = 1)
c_list_ml = c_list_ml+epil_columns+etio_columns
data_label.columns = c_list_ml
change_nan_list = data_label.columns
change_nan_list_interest = change_nan_list[60:92]
data_label = change_nan(data_label,change_nan_list_interest)
FO_label = data_label.iloc[idx_fo].reset_index(drop=True)
FO_true_label = FO_label['final outcome']
del FO_label['round'], FO_label['point outcome']
del FO_label['CBM'], FO_label['CBZ'], FO_label['LCS'], FO_label['LEV'], FO_label['LMT'], FO_label['NULL']
del FO_label['OXC'], FO_label['PER'], FO_label['PHT'], FO_label['TPM'], FO_label['VPA'], FO_label['ZNS'], FO_label['PGB']
FO_label.columns = FO_label.columns.astype(str)
categs_conti = list(FO_label.columns[1:4])+list(FO_label.columns[12:45]) 
change_nan_list_interest_fo = list(FO_label.columns[0:1])+list(FO_label.columns[4:12])+list(FO_label.columns[45:]) 
patient_total_list = FO_label['patient'].values