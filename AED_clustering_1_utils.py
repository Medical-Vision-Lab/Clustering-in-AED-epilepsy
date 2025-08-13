import sys
import copy
import statistics
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skimage
import tqdm
from numpy import arange
from scipy import stats
from scipy.stats import chi2_contingency, shapiro, ttest_ind, mannwhitneyu, pearsonr, f_oneway
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn import metrics, preprocessing, datasets
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    plot_confusion_matrix, f1_score, roc_curve, auc, roc_auc_score, make_scorer, silhouette_score
)
from sklearn.model_selection import (
    PredefinedSplit, KFold, StratifiedKFold, GridSearchCV,
    StratifiedShuffleSplit
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
from pyhelpers.store import save_svg_as_emf
from pyhelpers.dirs import cd
from gower import gower_matrix
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.stats.multitest import multipletests, pairwise_tukeyhsd
from pycirclize import Circos
import networkx as nx
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def performance_cal(pred, pred_prob, y_test):
    cm_rf = pd.DataFrame(confusion_matrix(y_test, pred), columns=['NSF', 'SF'], index=['NSF', 'SF'])
    TN = cm_rf.values[0, 0]
    FN = cm_rf.values[1, 0]
    TP = cm_rf.values[1, 1]
    FP = cm_rf.values[0, 1]
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    FDR = FP / (TP + FP)
    ACC = (TP + TN) / (TP + FP + FN + TN)
    F1_score = f1_score(y_true=y_test, y_pred=pred, pos_label=1)
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=pred_prob[:, 1], pos_label=1)
    AUC = auc(fpr, tpr)
    return [AUC, ACC, F1_score, TPR, TNR, PPV, NPV]

def experiment_AED_clustering(tr_AED, va_AED, te_AED, label_list, clf, cat_features):
    weight = compute_class_weight(class_weight="balanced",
                                  classes=np.unique(label_list[0]),
                                  y=label_list[0])
    weights = {0: weight[0], 1: weight[1]}

    c0_prior = (len(label_list[0][label_list[0] == 0]) / len(label_list[0]))
    c1_prior = (len(label_list[0][label_list[0] == 1]) / len(label_list[0]))

    scale_pos_weight = len(label_list[0][label_list[0] == 0]) / len(label_list[0][label_list[0] == 1])

    if clf == 'RF':
        param_grid = {'n_estimators': [100, 300, 500, 1000],
                      'max_depth': [6, 8, 10, 12],
                      'min_samples_leaf': [8, 12, 18],
                      'min_samples_split': [8, 16, 20]
                      }
        model = RandomForestClassifier(class_weight=weights)
    elif clf == 'XGB':
        param_grid = {'objective': ['binary:logistic'],
                      'metric': ['logloss'],
                      'eta': [0.05],
                      'boosting': ['gbdt', 'dart'],
                      'num_iterations': [1000],
                      'max_depth': [5, 7, 9],
                      'min_child_weight': [1, 3],
                      'n_estimators': [100, 300, 500, 1000],
                      'seed': [2018]}
        model = XGBClassifier(scale_pos_weight=scale_pos_weight)

    split_index = [-1] * len(tr_AED) + [0] * len(va_AED)
    X = np.concatenate((tr_AED, va_AED), axis=0)
    y = np.concatenate((label_list[0], label_list[1]), axis=0)
    pds = PredefinedSplit(test_fold=split_index)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=pds, scoring='roc_auc')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(te_AED)
    y_prob = best_model.predict_proba(te_AED)
    perfor = performance_cal(y_pred, y_prob, label_list[2])
    perfor.append(grid_search.best_score_)
    my_formatter = "{0:.3f}"
    cm_result = []
    for ii in range(len(perfor)):
        cm_result.append(my_formatter.format(perfor[ii]))
    return cm_result, y_pred, y_prob, label_list[2]

def strati_split(FO_label, rs=42):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
    for train_index, test_index in split.split(FO_label, FO_label["final outcome"]):
        FO_label_tr_va = FO_label.loc[train_index].sort_values(by=['patient']).reset_index(drop=True)
        FO_label_te = FO_label.loc[test_index].sort_values(by=['patient']).reset_index(drop=True)
    split_ = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=rs)
    for train_index, test_index in split_.split(FO_label_tr_va, FO_label_tr_va["final outcome"]):
        FO_label_tr = FO_label_tr_va.loc[train_index].sort_values(by=['patient']).reset_index(drop=True)
        FO_label_va = FO_label_tr_va.loc[test_index].sort_values(by=['patient']).reset_index(drop=True)
    return FO_label_tr, FO_label_va, FO_label_te

def clutering_compare(labels, selected_FO_true_label):
    label_no_minus = list(set(labels))
    while -1 in label_no_minus:
        label_no_minus.remove(-1)
    clustering_size = len(set(label_no_minus))
    cluster_sf = np.where(selected_FO_true_label == 1)
    cluster_nsf = np.where(selected_FO_true_label == 0)
    a_sf = []
    a_total = []
    a_observed = []
    a_pvalue = []
    for csize in label_no_minus:
        c_cluster = np.where(labels == csize)
        c_sf = np.intersect1d(c_cluster, cluster_sf).shape[0]/(np.intersect1d(c_cluster, cluster_sf).shape[0] + np.intersect1d(c_cluster, cluster_nsf).shape[0])
        c_total = c_cluster[0].shape[0]    
        c_observed = [np.intersect1d(c_cluster, cluster_sf).shape[0], np.intersect1d(c_cluster, cluster_nsf).shape[0]]
        a_sf.append(c_sf)
        a_total.append(c_total)
        a_observed.append(c_observed)
        
    c_observed = [cluster_sf[0].shape[0], cluster_nsf[0].shape[0]]
    a_observed.append(c_observed)
    
    for csize in range(clustering_size):
        observed = np.array([[a_observed[csize][0], a_observed[csize][1]], [a_observed[-1][0], a_observed[-1][1]]])
        chi2, p, dof, expected = chi2_contingency(observed)
        a_pvalue.append(p)
        
    return a_pvalue, a_observed
        
def showing_cluster_statistic_fin(labels, a_pvalue,selected_FO_label, selected_FO_true_label, save_name):
    a_size = np.where(np.array(a_pvalue) < 0.05)
    a_pvalue_whole = []
    final_feature = []
    compare_conti = 0
    compare_cate = 0
    for i in range(len(a_size[0])):
        final_feature_pre = []
        a_pvalue_column = []
        c_cluster = np.where(labels == a_size[0][i])
        cluster_df = selected_FO_label.loc[c_cluster]
        cluster_df['outcome'] = 1
        not_total = np.setdiff1d(selected_FO_label.index, c_cluster)
        comparison_df = selected_FO_label.loc[not_total]
        comparison_df['outcome'] = 0
        categorical_column = list(cluster_df.columns[0:1])+list(cluster_df.columns[4:10])+list(cluster_df.columns[45:84])
        conti_column = list(cluster_df.columns[1:4])+list(cluster_df.columns[10:45])
        
        total_df = pd.concat([cluster_df, comparison_df], axis = 0)
        df = pd.DataFrame(index=cluster_df.columns, columns=['type','p-value', 'total Num', 'summation','ratio', 'mean', 'sd', 'median', 'q1', 'q3', 'total Num c', 'summation c','ratio c', 'mean c', 'sd c', 'median c', 'q1 c', 'q3 c'])
        for column in cluster_df.columns:
            if column in conti_column:
                #print(0)
                group0_col = cluster_df[column]
                group1_col = comparison_df[column]
                _, p1 = shapiro(group0_col)
                _, p2 = shapiro(group1_col)
                alpha = 0.05
                if p2 > alpha:
                    # data is normally distributed, use t-test
                    t, p = ttest_ind(group0_col, group1_col)
                    df.loc[column, 'type'] = 'Normal'
                    df.loc[column, 'mean'] = np.mean(group0_col)
                    df.loc[column, 'sd'] = np.std(group0_col)
                    
                    df.loc[column, 'mean c'] = np.mean(group1_col)
                    df.loc[column, 'sd c'] = np.std(group1_col)
                    
                    #print(f'p-value for {column}: {p:.4f}')
                else:
                    # data is not normally distributed, use Mann-Whitney U test
                    stat, p = mannwhitneyu(group0_col, group1_col)
                    df.loc[column, 'type'] = 'Non-normal'
                    df.loc[column, 'median'] = np.std(group0_col)
                    df.loc[column, 'q1'] = np.percentile(group0_col, 25, interpolation='midpoint')
                    df.loc[column, 'q3'] = np.percentile(group0_col, 75, interpolation='midpoint')
                    
                    df.loc[column, 'median c'] = np.std(group1_col)
                    df.loc[column, 'q1 c'] = np.percentile(group1_col, 25, interpolation='midpoint')
                    df.loc[column, 'q3 c'] = np.percentile(group1_col, 75, interpolation='midpoint')
                    
                    #print(f'p-value for {column}: {p:.4f}')
                df.loc[column, 'p-value'] = p
                df.loc[column, 'total Num'] = len(group0_col)
                df.loc[column, 'total Num c'] = len(group1_col)
                a_pvalue_column.append(p)
            elif column in categorical_column:
                try:
                    group0_col = cluster_df[column]
                    group1_col = comparison_df[column]
                    
                    if column == 'Seizure classification':
                        group1_col = group1_col-1
                        group0_col = group0_col-1
                        
                    observed = np.array([[np.sum(group0_col), len(group0_col)-np.sum(group0_col)],[np.sum(group1_col), len(group1_col)-np.sum(group1_col)]])
                    chi2, p, dof, expected = chi2_contingency(observed)
                    a_pvalue_column.append(p)
                    df.loc[column, 'type'] = 'Cate'
                    df.loc[column, 'p-value'] = p
                    df.loc[column, 'total Num'] = len(group0_col)
                    df.loc[column, 'summation'] = np.sum(group0_col)
                    df.loc[column, 'ratio'] = np.sum(group0_col)/len(group0_col)
                    
                    df.loc[column, 'total Num c'] = len(group1_col)
                    df.loc[column, 'summation c'] = np.sum(group1_col)
                    df.loc[column, 'ratio c'] = np.sum(group1_col)/len(group1_col)
                except:
                    group0_col = cluster_df[column]
                    group1_col = comparison_df[column]
                    
                    if column == 'Seizure classification':
                        group1_col = group1_col-1
                        group0_col = group0_col-1
                        
                    observed = np.array([[np.sum(group0_col), len(group0_col)-np.sum(group0_col)],[np.sum(group1_col), len(group1_col)-np.sum(group1_col)]])
                    chi2, p, dof, expected = 1, 1, 1, 1
                    a_pvalue_column.append(p)
                    df.loc[column, 'type'] = 'wrong'
                    df.loc[column, 'p-value'] = 1
                    df.loc[column, 'total Num'] = 1
                    df.loc[column, 'summation'] = 1
                    df.loc[column, 'ratio'] = 1
                    df.loc[column, 'total Num c'] = 1
                    df.loc[column, 'summation c'] = 1
                    df.loc[column, 'ratio c'] = 1
            # multi-variate analysis 

        df_name = 'df_' +str(i)+'.csv'
        cluster_name = 'cluster_' +str(i)+'.csv'
        df.to_csv(df_name)
        cluster_df.to_csv(cluster_name)
        comparison_name = 'comparison_' +str(i)+'.csv'
        comparison_df.to_csv(comparison_name)
    return a_pvalue_whole

def remove_CNZ(PO_tr_dat):
    PO_tr_dat = PO_tr_dat.drop(['CNZ'], axis=1)
    idx_remov = []
    for ii in range(len(PO_tr_dat)):
        if sum(PO_tr_dat.iloc[ii, 4:16]) == 0:
            idx_remov.append(ii)
    PO_tr_dat = PO_tr_dat.drop(idx_remov).reset_index(drop=True)
    print('removed case:',str(len(idx_remov)))
    return PO_tr_dat

def issue_yes_no(label_PO_te):
    label_PO_te['wave_issue'] = label_PO_te['Wave_RSW_G']+label_PO_te['Wave_SW_G']+label_PO_te['Wave_RDA_G']+\
        label_PO_te['Wave_PD_G']+label_PO_te['Wave_RSW_NG']+label_PO_te['Wave_SW_NG']+label_PO_te['Wave_RDA_NG']+\
        label_PO_te['Wave_PD_NG']
    label_PO_te['slow_issue'] = label_PO_te['Slow_CS_G']+label_PO_te['Slow_IS_G']+label_PO_te['Slow_CS_NG']+label_PO_te['Slow_IS_NG']
    label_PO_te['mri_issue'] = label_PO_te[1]+label_PO_te[2]+label_PO_te[3]+\
        label_PO_te[4]+label_PO_te[5]+label_PO_te[6]+label_PO_te[7]+\
        label_PO_te[8]+label_PO_te[9]+label_PO_te[10]+label_PO_te[11]+\
        label_PO_te[12]+label_PO_te[13]+label_PO_te[14]+label_PO_te[15]+\
        label_PO_te[16]+label_PO_te[99]
    for patient_idx in range(label_PO_te.shape[0]):
        if label_PO_te['wave_issue'][patient_idx] > 0 :
            label_PO_te['wave_issue'][patient_idx] = 1
        elif label_PO_te['wave_issue'][patient_idx] < 0:
            label_PO_te['wave_issue'][patient_idx] = np.nan
        
        if label_PO_te['slow_issue'][patient_idx] > 0 :
            label_PO_te['slow_issue'][patient_idx] = 1
        elif label_PO_te['slow_issue'][patient_idx] < 0:
            label_PO_te['slow_issue'][patient_idx] = np.nan
            
        if label_PO_te['mri_issue'][patient_idx] > 0 :
            label_PO_te['mri_issue'][patient_idx] = 1
        elif label_PO_te['mri_issue'][patient_idx] < 0:
            label_PO_te['mri_issue'][patient_idx] = np.nan
            
    return label_PO_te

def make_dat_label(FO_tr_dat):
    tr_label = FO_tr_dat['final outcome'].values
    tr_data = FO_tr_dat.iloc[:,18:].values.astype(object)
    return tr_data, tr_label
  
def isNaN(string):
    return not (string != string)

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

def plot_cluster_connections(df, feature_group, title):
    feature_group_array = np.array(feature_group)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    angles = np.linspace(0, 2 * np.pi, len(clusters), endpoint=False).tolist()
    colors = plt.cm.get_cmap('tab10', len(feature_group))
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (3, 10, 1, 10)),
                   (0, (5, 10)), 'solid', (0, (1, 10))] * (len(feature_group) // 10 + 1)  
    color_map = {feature: colors(i) for i, feature in enumerate(feature_group)}
    line_style_map = {feature: line_styles[i] for i, feature in enumerate(feature_group)}
    for _, row in df.iterrows():
        feature = row['Feature']
        color = color_map[feature]
        line_style = line_style_map[feature]

        start_cluster = clusters.index(row['group1'])
        end_cluster = clusters.index(row['group2'])
        if feature in feature_group_array:
            feature_index = np.where(feature_group_array == feature)[0][0
        else:
            feature_index = 0 s
        offset_scale = 0.2
        offset = offset_scale * ((feature_index - len(feature_group_array) / 2) / len(feature_group_array))
        start_angle = angles[start_cluster] + offset
        end_angle = angles[end_cluster] + offset
        ax.plot([start_angle, end_angle], [1, 1], color=color, linestyle=line_style, linewidth=2, alpha=0.7, label=feature)
    for angle, cluster in zip(angles, clusters):
        ax.plot(angle, 1, 'o', markersize=100)  # Removed labels here to avoid redundancy
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right', bbox_to_anchor=(1.2, 1.2))
    plt.axis('off')
    plt.show()

def shift_row(row):
    valid_indices = np.where(~row.isna())[0]
    if len(valid_indices) == 0:
        return row  # If all values are NaN, return the row as is
    first_valid_index = valid_indices[0]
    shifted_row = row[first_valid_index:].reset_index(drop=True)
    shifted_row = shifted_row.reindex(range(len(row)))
    return shifted_row

def interpolate_row(row):
    last_valid_index = row.last_valid_index()
    if last_valid_index is None:
        return row  # If the entire row is NaN, return it as is
    row[:last_valid_index+1] = row[:last_valid_index+1].interpolate()
    return row

def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    is_outlier = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
    return df[~is_outlier.any(axis=1)]

def remove_correaltion(df):
    correlation_matrix = df.T.corr().abs()
    average_correlation = correlation_matrix.mean(axis=1)
    patients_to_keep = average_correlation[average_correlation > 0.1].index
    filtered_data_high_correlation = df.loc[patients_to_keep]
    return filtered_data_high_correlation

def remove_outliers_zscore(df, threshold=3):
    z_scores = np.abs(zscore(df, nan_policy='omit'))
    row_outliers = np.any(z_scores > threshold, axis=1)
    return df[~row_outliers]