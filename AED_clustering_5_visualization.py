csv_files = [
    'cluster_0_csv', 
    'cluster_1.csv', 
    'cluster_2.csv', 
    'cluster_3.csv', 
    'cluster_4.csv'
] # You can get each cluster df using "AED_clustering_4_clustering.py"

dfs = []
path = 'save_path'
for file in csv_files:
    df = pd.read_csv(path + file)
    dfs.append(df)
for i, df in enumerate(dfs):
    df['cluster'] = f'Cluster {i + 1}'
merged_df = pd.concat(dfs, ignore_index=True)
numeric_df = merged_df.select_dtypes(include=[np.number]) 
corr_matrix = numeric_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
reduced_df = merged_df.drop(columns=to_drop)
numeric_df = merged_df.select_dtypes(include=[np.number])
while True:
    vif_data = calculate_vif(numeric_df)
    max_vif = vif_data['VIF'].max()
    if max_vif > 5:
        feature_to_remove = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
        print(f"Removing: {feature_to_remove} with VIF: {max_vif}")
        numeric_df = numeric_df.drop(columns=[feature_to_remove])
    else:
        break
reduced_df = numeric_df.join(merged_df['cluster'])
p_values = []
features = reduced_df.columns[1:-1]

for feature in features:
    groups = [merged_df[feature][merged_df['cluster'] == f'Cluster {i+1}'] for i in range(len(csv_files))]
    stat, p_value = f_oneway(*groups)
    p_values.append(p_value)
    print(p_value)
results = pd.DataFrame({
    'Feature': features,
    'p-value': p_values
})
results['p-value'] = results['p-value'].apply(
    lambda x: np.random.uniform(0.1, 0.5) if pd.isna(x) else x
)
results['FDR'] = multipletests(results['p-value'], method='fdr_bh')[1]
significant_features = results[results['FDR'] < 0.01]
for feature in significant_features['Feature']:
    tukey = pairwise_tukeyhsd(endog=merged_df[feature], groups=merged_df['cluster'], alpha=0.05)
tukey_results = []
for feature in significant_features['Feature']:
    tukey = pairwise_tukeyhsd(endog=merged_df[feature], groups=merged_df['cluster'], alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df['Feature'] = feature 
    tukey_results.append(tukey_df)
final_tukey_results = pd.concat(tukey_results, ignore_index=True)
final_tukey_results.to_csv('tukey_hsd_results.csv', index=False)
final_tukey_results = pd.read_csv('tukey_hsd_results.csv')
final_tukey_results = final_tukey_results[final_tukey_results['p-adj'] < 0.01]
final_tukey_results = final_tukey_results.dropna()
clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
features = final_tukey_results['Feature'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
color_map = dict(zip(features, colors))
angles = np.linspace(0, 2 * np.pi, len(clusters), endpoint=False).tolist()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
for angle, cluster in zip(angles, clusters):
    ax.plot(angle, 1, 'o', label=cluster, markersize=50) 
offset_scale = 0.2
for _, row in final_tukey_results.iterrows():
    feature = row['Feature']
    color = color_map[feature]
    start_cluster = clusters.index(row['group1'])
    end_cluster = clusters.index(row['group2'])
    feature_index = np.where(features == feature)[0][0]
    offset = offset_scale * ((feature_index - len(features) / 2) / len(features))
    start_angle = angles[start_cluster] + offset
    end_angle = angles[end_cluster] + offset
    ax.plot([start_angle, end_angle], [1, 1], color=color, linewidth=2, alpha=0.7)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2))
plt.show()

feature_groups = {
    'group1': [list(features[0:3])][0],
    'group2': [list(features[3:7])][0],
    'group3': [list(features[7:])][0]
}

for key, group_features in feature_groups.items():
    filtered_df = final_tukey_results[final_tukey_results['Feature'].isin(group_features)]
    plot_cluster_connections(filtered_df, group_features, f'Connections for {key}')
    
G = nx.Graph()
nodes = range(1, 6)
G.add_nodes_from(nodes)
edges = []
for _, row in final_tukey_results.iterrows():
    feature = row['Feature']
    edges.append((int(row['group1'][-1]), int(row['group2'][-1]), {'feature': row['Feature'], 'weight':3},))
G.add_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=700)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
labels = nx.get_edge_attributes(G, 'feature')
weighted_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['feature']]
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
plt.axis('off') 
plt.show()
FO_label_with_clusters = FO_label.copy()
FO_label_with_clusters['cluster_label'] = np.nan
FO_label_with_clusters.loc[clustering_set, 'cluster_label'] = labels_for_identify
file_path = 'longitudinal_outcome.xlsx'
df = pd.read_excel(file_path, sheet_name=1)
del df['연번']
shifted_df = df.apply(shift_row, axis=1)
first_row = shifted_df.iloc[2584]
plt.figure(figsize=(12, 6))
plt.plot(first_row.index, first_row.values, marker='o')
plt.title('Plot of the First Row')
plt.xlabel('Column Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
interpolated_df = shifted_df.apply(interpolate_row, axis=1)
first_row = interpolated_df.iloc[2584].dropna()
plt.figure(figsize=(12, 6))
plt.plot(first_row.index, first_row.values, marker='o')
plt.title('Plot of the First Row after Interpolation')
plt.xlabel('Column Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

FO_label_patient = pd.read_csv('FO_label_patient.csv')['patient']-1
FO_label_patient = FO_label_patient.values
FO_df = interpolated_df.iloc[FO_label_patient]
FO_df = FO_df.reset_index()
del FO_df['index']
FO_label_cluster = pd.read_csv('FO_label_cluster.csv')['0'].values
FO_cl = FO_df.iloc[FO_label_cluster]
FO_cl = FO_cl.reset_index()
del FO_cl['index']
for cluster_i in range(5):
    FO_label_each = pd.read_csv(f'cluster_{cluster_i}.csv')['Unnamed: 0'].values
    FO_cl_each = FO_cl.iloc[FO_label_each]
    FO_cl_each = FO_cl_each.reset_index()
    del FO_cl_each['index']
    plt.figure(figsize=(12, 6))
    for i in range(FO_cl_each.shape[0]):
        plt.plot(FO_cl_each.iloc[i].dropna(), alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.close()
    mean_values_interpolate = FO_cl_each.mean()
    FO_cl_each_filled = FO_cl_each.apply(lambda x: x.fillna(mean_values_interpolate[x.name]), axis=0)
    mean_values = FO_cl_each_filled.mean()
    std_values = FO_cl_each_filled.std()
    mean_values = pd.to_numeric(mean_values, errors='coerce')
    std_values = pd.to_numeric(std_values, errors='coerce')
    mean_values = np.nan_to_num(mean_values, nan=0.0, posinf=0.0, neginf=0.0)
    std_values = np.nan_to_num(std_values, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = mean_values - std_values
    maximum = mean_values + std_values
    plt.figure(figsize=(12, 6))
    plt.plot(mean_values, label='Mean', color='blue')
    plt.fill_between(np.arange(52), minimum, maximum, color='blue', alpha=0.2, label='Std Dev')
    plt.title('Mean and Standard Deviation of Rows')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    exec(f'mean_values_{cluster_i} = mean_values')

for cluster_i in range(5):
    FO_label_each = pd.read_csv(f'cluster_{cluster_i}.csv')['Unnamed: 0'].values
    FO_cl_each = FO_cl.iloc[FO_label_each]
    FO_cl_each = FO_cl_each.reset_index()
    del FO_cl_each['index']
    plt.figure(figsize=(12, 6))
    for i in range(FO_cl_each.shape[0]):
        plt.plot(FO_cl_each.iloc[i].dropna(), alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.close()
    filtered_columns = FO_cl_each.columns[FO_cl_each.notna().sum() >= int(FO_cl_each.shape[0]*0.1)]
    filtered_data = FO_cl_each[filtered_columns]
    plt.figure(figsize=(12, 6))
    for i in range(filtered_data.shape[0]):
        plt.plot(filtered_data.iloc[i].dropna(), alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.close()
    filtered_data_no_outliers = remove_outliers_iqr(filtered_data)
    plt.figure(figsize=(12, 6))
    for i in range(filtered_data_no_outliers.shape[0]):
        plt.plot(filtered_data_no_outliers.iloc[i].dropna(), alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.close()
    filtered_data_high_correlation = remove_correaltion(filtered_data)
    plt.figure(figsize=(12, 6))
    for i in range(filtered_data_high_correlation.shape[0]):
        plt.plot(filtered_data_high_correlation.iloc[i].dropna(), alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    filtered_data_no_outliers = remove_outliers_zscore(filtered_data_high_correlation)
    plt.figure(figsize=(12, 6))
    for i in range(filtered_data_no_outliers.shape[0]):
        plt.plot(filtered_data_no_outliers.iloc[i].dropna(), alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    FO_cl_each = filtered_data_no_outliers.copy()
    mean_values_interpolate = FO_cl_each.mean()
    FO_cl_each_filled = FO_cl_each.apply(lambda x: x.fillna(mean_values_interpolate[x.name]), axis=0)
    mean_values = FO_cl_each_filled.mean()
    std_values = FO_cl_each_filled.std()
    mean_values = pd.to_numeric(mean_values, errors='coerce')
    std_values = pd.to_numeric(std_values, errors='coerce')
    mean_values = np.nan_to_num(mean_values, nan=0.0, posinf=0.0, neginf=0.0)
    std_values = np.nan_to_num(std_values, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = mean_values - std_values
    maximum = mean_values + std_values
    plt.figure(figsize=(12, 6))
    plt.plot(mean_values, label='Mean', color='blue')
    plt.fill_between(np.arange(minimum.shape[0]), minimum, maximum, color='blue', alpha=0.2, label='Std Dev')
    plt.title('Mean and Standard Deviation of Rows')
    plt.xlabel('Qarter')
    plt.ylabel('Seizure frequency')
    plt.legend()
    plt.grid(True)
    plt.close()
    exec(f'mean_values_{cluster_i} = mean_values')
    plt.figure(figsize=(12, 6))
    for i in range(FO_cl_each_filled.shape[0]):
        plt.plot(FO_cl_each.iloc[i].dropna(), alpha=0.5)
    plt.xlabel('Qarter')
    plt.ylabel('Seizure frequency')
    plt.grid(True)
    plt.title(f'cluster_{cluster_i}')
    plt.savefig(f"cluster_{cluster_i}.svg",dpi=700) #.png,.pdf will also support here
    plt.close()

plt.figure(figsize=(12, 6))
for cluster_i in range(5):
    exec(f'mean_values = mean_values_{cluster_i}')
    plt.plot(mean_values, label=f'cluster {cluster_i}')
    plt.xlabel('Qarter')
    plt.ylabel('Seizure frequency')
    plt.legend()
plt.savefig("E:/Epilepsy_dataset/Tabular/result_clustering_240708/Avg_cluter.svg",dpi=700) 
plt.close()

for cluster_i in range(5):
    FO_label_each = pd.read_csv(f'cluster_{cluster_i}.csv')['Unnamed: 0'].values
    FO_cl_each = FO_cl.iloc[FO_label_each]
    FO_cl_each = FO_cl_each.reset_index()
    del FO_cl_each['index']
    plt.figure(figsize=(12, 6))
    for i in range(FO_cl_each.shape[0]):
        plt.plot(FO_cl_each.iloc[i].dropna(), alpha=0.3)
    plt.title('Overlapping Plot of 50 Lines with Alpha 0.3')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.close()
    mean_values_interpolate = FO_cl_each.mean()
    FO_cl_each_filled = FO_cl_each.apply(lambda x: x.fillna(mean_values_interpolate[x.name]), axis=0)
    mean_values = FO_cl_each_filled.mean()
    std_values = FO_cl_each_filled.std()
    mean_values = pd.to_numeric(mean_values, errors='coerce')
    std_values = pd.to_numeric(std_values, errors='coerce')
    mean_values = np.nan_to_num(mean_values, nan=0.0, posinf=0.0, neginf=0.0)
    std_values = np.nan_to_num(std_values, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = mean_values - std_values
    maximum = mean_values + std_values
    plt.figure(figsize=(12, 6))
    plt.plot(mean_values, label='Mean', color='blue')
    plt.fill_between(np.arange(52), minimum, maximum, color='blue', alpha=0.2, label='Std Dev')
    plt.title('Mean and Standard Deviation of Rows')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    exec(f'mean_values_{cluster_i} = mean_values')

df0 = np.arange(52)
data = {
    'time': df0,
    'event': mean_values_1
}
df1 = pd.DataFrame(data)
data = {
    'time': df0,
    'event': mean_values_2
}
df2 = pd.DataFrame(data)
kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()
plt.figure(figsize=(12, 6))
kmf1.fit(df1['time'], event_observed=df1['event'], label='Group 1')
kmf1.plot_survival_function()
kmf2.fit(df2['time'], event_observed=df2['event'], label='Group 2')
kmf2.plot_survival_function()
plt.title('Survival Curves')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()
results = logrank_test(df1['time'], df2['time'], event_observed_A=df1['event'], event_observed_B=df2['event'])
print(f'Log-Rank Test p-value: {results.p_value}')
if results.p_value < 0.05:
    print("The difference between the two groups is statistically significant.")
else:
    print("The difference between the two groups is not statistically significant.")
data = {
    'group1': {
        'time': df0,
        'event': mean_values_0
    },
    'group2': {
        'time': df0,
        'event': mean_values_1
    },
    'group3': {
        'time': df0,
        'event': mean_values_2
    },
    'group4': {
        'time': df0,
        'event': mean_values_3
    },
    'group5': {
        'time': df0,
        'event': mean_values_4
    }
}
dfs = {group: pd.DataFrame(data[group]) for group in data}
plt.figure(figsize=(12, 6))
kmf = KaplanMeierFitter()
for group, df in dfs.items():
    kmf.fit(df['time'], event_observed=df['event'], label=group)
    kmf.plot_survival_function()

plt.title('Survival Curves')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.legend()
plt.show()

data = {
    'group1': mean_values_0,
    'group2': mean_values_1,
    'group3': mean_values_2,
    'group4': mean_values_3,
    'group5': mean_values_4
}
dfs = {group: pd.DataFrame({'pain': data[group]}) for group in data}
df_combined = pd.concat([df.assign(group=group) for group, df in dfs.items()])
plt.figure(figsize=(12, 6))
sns.boxplot(x='group', y='pain', data=df_combined)
plt.title('Pain Severity Distributions Across Groups')
plt.xlabel('Group')
plt.ylabel('Pain Severity')
plt.grid(True)
plt.show()