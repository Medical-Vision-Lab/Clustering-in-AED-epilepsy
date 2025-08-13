patient_corr = np.zeros(shape = (len(patient_total_list),100))
patient_corr_db = pd.DataFrame(patient_corr.T)
patient_corr_db.columns = patient_total_list
patient_entire_db = pd.DataFrame(patient_corr.T)
patient_entire_db.columns = patient_total_list
for rs in tqdm.trange(0,100):
    label_FO_tr, label_FO_va, label_FO_te = strati_split(FO_label, rs)
    FO_ratio_tr = np.histogram(label_FO_tr['final outcome'],len(set(label_FO_tr['final outcome'])))
    FO_ratio_va = np.histogram(label_FO_va['final outcome'],len(set(label_FO_va['final outcome'])))
    FO_ratio_te = np.histogram(label_FO_te['final outcome'],len(set(label_FO_te['final outcome'])))
    categs_conti = list(label_FO_tr.columns[18:21])+list(label_FO_tr.columns[27:60]) # continuous value 
    tr_ratio = np.histogram(label_FO_tr['final outcome'],len(set(label_FO_tr['final outcome'])))
    va_ratio = np.histogram(label_FO_va['final outcome'],len(set(label_FO_va['final outcome'])))
    te_ratio = np.histogram(label_FO_te['final outcome'],len(set(label_FO_te['final outcome'])))
    tr_label= label_FO_tr['final outcome']
    va_label= label_FO_va['final outcome']
    te_label= label_FO_te['final outcome']
    test_patient = label_FO_te['patient'].values
    tr_data = label_FO_tr.iloc[:,2:].values.astype(object) 
    va_data = label_FO_va.iloc[:,2:].values.astype(object)
    te_data = label_FO_te.iloc[:,2:].values.astype(object)
    c_List_FO = label_FO_tr.columns[2:]
    cat_features = [0] + list(range(4, 9)) + list(range(43, 84))
    for ii in cat_features:
        tr_data[:,ii] = tr_data[:,ii].astype(int)
        va_data[:, ii] = va_data[:, ii].astype(int)
        te_data[:, ii] = te_data[:, ii].astype(int)
    fo_XGB, pred_pre, prob_pre, label_pre = experiment_AED_clustering(tr_data, va_data, te_data, [tr_label, va_label, te_label], 'XGB', cat_features)
    correct_tp = test_patient[np.where(pred_pre == label_pre)]
    for ci in correct_tp:
        patient_corr_db[ci][rs] = patient_corr_db[ci][rs]+1
        
patient_corr_db_mean = np.mean(np.array(patient_corr_db), axis = 0)
plt.bar(np.arange(len(patient_corr_db_mean)),patient_corr_db_mean)
for rs in tqdm.trange(0,100):
    label_FO_tr, label_FO_va, label_FO_te = strati_split(FO_label, rs)
    FO_ratio_tr = np.histogram(label_FO_tr['final outcome'],len(set(label_FO_tr['final outcome'])))
    FO_ratio_va = np.histogram(label_FO_va['final outcome'],len(set(label_FO_va['final outcome'])))
    FO_ratio_te = np.histogram(label_FO_te['final outcome'],len(set(label_FO_te['final outcome'])))
    categs_conti = list(label_FO_tr.columns[18:21])+list(label_FO_tr.columns[27:60]) # continuous value 
    tr_ratio = np.histogram(label_FO_tr['final outcome'],len(set(label_FO_tr['final outcome'])))
    va_ratio = np.histogram(label_FO_va['final outcome'],len(set(label_FO_va['final outcome'])))
    te_ratio = np.histogram(label_FO_te['final outcome'],len(set(label_FO_te['final outcome'])))
    tr_label= label_FO_tr['final outcome']
    va_label= label_FO_va['final outcome']
    te_label= label_FO_te['final outcome']
    test_patient = label_FO_te['patient'].values
    for ci in test_patient:
        patient_entire_db[ci][rs] = patient_entire_db[ci][rs]+1

patient_entire_db_mean = np.mean(np.array(patient_entire_db), axis = 0)
best_patient_group = patient_corr_db_mean/patient_entire_db_mean
best_patient_group_where = np.where(best_patient_group>0.5)
interest = FO_label.columns
FO_true_label = FO_label['final outcome']
selected_FO_label = FO_label.copy()
selected_FO_label = selected_FO_label.iloc[np.where(best_patient_group_where[0])]
selected_FO_true_label = selected_FO_label['final outcome']
del FO_label['final outcome'], FO_label['patient']
del selected_FO_label['patient'], selected_FO_label['final outcome']
categs_conti = list(selected_FO_label.columns[1:4])+list(selected_FO_label.columns[10:43]) # continuous value 
for col in categs_conti:
    selected_FO_label[col] = selected_FO_label[col].fillna(selected_FO_label.loc[:, col].median())

selected_gower_mat = gower_matrix(selected_FO_label)
selected_gower_mat = pd.DataFrame(selected_gower_mat, index=selected_FO_label.index, columns=selected_FO_label.index)
        
perplexity_p, learning_rate_p, early_exaggeration_p = 50, 200, 400
tsne = TSNE(n_components=2, perplexity=perplexity_p, learning_rate = learning_rate_p, random_state=0, early_exaggeration = early_exaggeration_p)
selected_tsne_mat = tsne.fit_transform(selected_gower_mat)
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.Set1(range(2))
for i in range(2):
    ax.scatter(selected_tsne_mat[selected_FO_true_label==i, 0], selected_tsne_mat[selected_FO_true_label==i, 1], c=colors[i], label=f'Cluster {i}')
ax.legend()
plt.title(str(perplexity_p)+'   '+ str(learning_rate_p)+'   '+ str(early_exaggeration_p))
save_name = 'D:/clustering/2d_wo_selection' + str(perplexity_p)+'_'+ str(learning_rate_p)+'_'+ str(early_exaggeration_p)+'.png'
plt.savefig(save_name)
plt.close()
cluster_size_good = False
initial_min_size = 5
while True:
    hdbscn = hdbscan.HDBSCAN(min_cluster_size=initial_min_size)
    labels = hdbscn.fit_predict(selected_tsne_mat)
    labels_for_identify = labels.copy()
    hdbscan_pvalue, hdbscn_observed = clutering_compare(labels, selected_FO_true_label)
    if len(hdbscn_observed) < 5 or (5 < len(hdbscn_observed) < 15):
        break
    initial_min_size += 1

hdbscan_pvalue_column = showing_cluster_statistic_fin4(labels, hdbscan_pvalue,selected_FO_label, selected_FO_true_label, save_name)
fig, ax = plt.subplots(figsize=(8, 6))
labels_list = list(set(labels))
colors = plt.cm.Set1(np.linspace(0, 1, len(labels_list)))
colors2 = np.array([[0.89411765, 0.10196078, 0.10980392, 1.        ],
       [0.7, 0.6, 0.6, 1.        ],
       [0.8, 0.6, 0.6, 1.        ],
       [0.9, 0.6, 0.6, 1.        ],
       [0.5, 0.6, 0.6, 1.        ],
       [0.4, 0.6, 0.6, 1.        ],
       [0.3, 0.6, 0.6, 1.        ],
       [0.2, 0.6, 0.6, 1.        ],
       [0.21568627, 0.49411765, 0.72156863, 1.        ],
       [0.30196078, 0.68627451, 0.29019608, 1.        ],
       [0.59607843, 0.30588235, 0.63921569, 1.        ],
       [0.1, 0.6, 0.6, 1.        ],
       [0.96862745, 0.50588235, 0.74901961, 1.        ], 
       [0.0, 0.6, 0.6, 1.        ]])
colors = colors2
for i, label in enumerate(labels_list):
    if label == -1:        
        color = 'black'
        cluster_label = 'Noise'
    else:
        color = colors[i]
        cluster_label = f'Cluster {label+1}  [{hdbscn_observed[label][0]},{hdbscn_observed[label][1]}] {hdbscan_pvalue[label]:.3f}'
        
    scatter = ax.scatter(selected_tsne_mat[labels == label, 0], selected_tsne_mat[labels == label, 1],
                         c=[color], label=cluster_label)
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles)) 
plt.legend(unique_labels.values(), unique_labels.keys())
plt.xlabel('Component 0')
plt.ylabel('Component 1')
plt.title('HDBSCAN Clustering with t-SNE')
plt.title(str(perplexity_p)+'   '+ str(learning_rate_p)+'   '+ str(early_exaggeration_p))
save_name = 'Clustering.svg'
save_name2 = 'Clustering.emf'
plt.savefig(save_name,dpi=700) #.png,.pdf will also support here
svg_file_pathname = cd(save_name)
emf_file_pathname = cd(save_name2)
save_svg_as_emf(svg_file_pathname, emf_file_pathname, verbose=True)
plt.close()
    