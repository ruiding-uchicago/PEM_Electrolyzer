import numpy as np
import pandas as pd
def find_inter_value(feature1,feature2,inter_frame):
    for i in range(0,inter_frame.shape[0]):
        if feature1 in inter_frame.loc[i]['Feature_1']:
            if feature2 in inter_frame.loc[i]['Feature_2']:
                return (inter_frame.loc[i]['Interaction'])
def find_the_interaction_matrix(inter_data_frame,feature_list):
    result_matrix=pd.DataFrame(index=feature_list,columns=feature_list)
    for feature_1 in feature_list:
        for feature_2 in feature_list:
            result_matrix.loc[feature_1][feature_2]=find_inter_value(feature_1,feature_2,inter_data_frame)
    for feature_1 in feature_list:
        for feature_2 in feature_list:
            if result_matrix.loc[feature_1][feature_2]==None:
                result_matrix.loc[feature_1][feature_2]=result_matrix.loc[feature_2][feature_1]
    return result_matrix
def df_norm(target_dataframe):
    normed_df = (target_dataframe - target_dataframe.min().min()) / (target_dataframe.max().max() - target_dataframe.min().min())
    return normed_df
def process_matrix(file,feature_list):
    data=pd.read_csv(file,encoding='gbk', names=["Feature_1", "Feature_2", "Interaction"])
    result_matrix=find_the_interaction_matrix(data,feature_list)
    result_matrix=df_norm(result_matrix)
    result_matrix=result_matrix.fillna(0)
    result_matrix.to_csv('Matrix'+file)
    return result_matrix
