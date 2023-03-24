import pandas as pd
import numpy as np
         
def binning_bmi(data):
    """
    Binning Body Mass Index (BMI)
    """
    # underweight
    group = 1
    # healthy weight
    if 18.5 <= data < 25:
        group = 2
    # overweight
    if 25 <= data < 30:
        group = 3
    # obesity
    if data >= 30:
        group = 4
        
    return group

def binning_sleep_duration(data):
    """
    Binning sleep duration (hours)
    """
    # less than 7 hours
    group = 1
    # 7-9 hours (recommended)
    if 7 <= data < 9:
        group = 2
    # more than 9 hours
    if data > 9:
        group = 3
        
    return group

def binning_data(data):
    """
    Binning BMI and sleep duration
    """
    data['BMI Group'] = data['BMI'].apply(binning_bmi)
    data['Sleep Duration Group'] = data['Sleep Duration (Hours)'].apply(binning_sleep_duration)
    data = data.drop(columns=['BMI', 'Sleep Duration (Hours)'])

    return data

def one_hot_encode(data):
    """
    Method for One-Hot Encoding 
    """
    cate_list = list(data.select_dtypes(include=['category', 'object']).columns)
    df_encoded = pd.get_dummies(data, columns=cate_list, prefix_sep='_')
    # drop columns end with '_No'
    df_encoded = df_encoded[df_encoded.columns.drop(list(df_encoded.filter(regex='_No$')))]
    # remove '_Yes', 'Gender_', and 'Race_' from column names
    df_encoded.columns = df_encoded.columns.str.replace("_Yes|Gender_|Race_", "", regex=True)
    # drop redundant columns to reduce the impact of multicollinearity
    df_encoded = df_encoded.drop(columns=['Male', 'Other Race - Including Multi-Racial'], errors='ignore')

    return df_encoded

def oversampling(X, y, oversampler, sampling_strategy="auto"):
    """Oversampling the minority class
    
    Args:
        X: independent variables
        y: corresponding target variable
        oversampler: oversampling method that will be applied to X, y 
        
    Returns:
        X and y after oversampling
    """
    model = oversampler(random_state=42, sampling_strategy=sampling_strategy)
    X_oversample, y_oversample = model.fit_resample(X, y)

    return X_oversample, y_oversample

def standardize(data, save_csv=False, file_path=""):
    data_mean = data.mean()
    data_std = data.std()
    data_scaled = (data - data_mean)/data_std
    # save mean and std as .csv
    if save_csv and file_path:
        df_mean = data_mean.to_frame().T
        df_std = data_std.to_frame().T
        combined = pd.concat([df_mean, df_std])
        combined.to_csv(file_path, index=False)

    return data_scaled