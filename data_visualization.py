import pandas as pd
import streamlit as st
import plotly.express as px
from constants import *

NUM_CATE_MAPPING_DICT = {
    'Age':  {
        0: "40-44",
        45: "45-49", 
        50: "50-54",
        55: "55-59",
        60: "60-64", 
        65: "65-69",
        70: "70-74",
        75: "75-79", 
        80: "80+",
    },
    'BMI': {
        0: "Underweight",
        18.5: "Healthy Weight",
        25: "Overweight",
        30: "Obesity"
    },
    'Sleep Duration (Hours)': {
        0: "Less than 7 Hours",
        7: "7-9 Hours",
        9: "More than 9 Hours",
    }  
}

# Global variables for UI
page_title = "Osteoporosis Prevalence"
info_text = """
            This page shows osteoporosis prevalence of the National Health and Nutrition Examination Survey (NHANES) data for years 2013-2014, and 2017-2020, among 5831 respondents aged 40-80 (and plus).
        """

def get_prevalence(data, feature, target):
    """Get the percentage of 'Yes'"""

    df = data.copy()     
    df = df.groupby([feature, target]).size().unstack(fill_value=0).reset_index()
    df['Prevalence'] = round(df["Yes"]/(df["Yes"]+df["No"])*100, 2)
    
    return df

def number_to_category(data, feature, new_feature_group):
    """Group by a specific variable based on pre-defined dictionary
    
    Args:
        data: the input dataframe
        feature: the numeric variable to be grouped
        new_feature_group: the new variable with data type string
    Returns:
        a dataframe in which the numeric variable has been converted to strings
    """

    df = data.copy()
    if feature in NUM_CATE_MAPPING_DICT:
        # find the corresponding dict
        threshold_dict = NUM_CATE_MAPPING_DICT[feature]
        for threshold_value, new_value in threshold_dict.items():
            df.loc[df[feature] >= threshold_value, new_feature_group] = new_value 
     
    return df

def barplot_prevalence(data, feature, tickangle=0, categoryarray=None):
    """Create bar charts of prevalence
    
    Args:
        data: the input dataframe
        feature: the variable that shown on x axis
        tickangle: the angle of rotation of x-axis tick labels
        categoryarray: the custom order of x-axis tick labels
    """

    # get prevalence of each group of x
    df_prevalence = get_prevalence(data, feature=feature, target=TARGET_NAME)

    # create bar charts
    fig = px.bar(df_prevalence, x=feature, y="Prevalence", color=feature, text='Prevalence', title=f'Prevalence By {feature}')
    fig.update_layout(showlegend=False,
                      yaxis_title='Percent',
                     )
    fig.update_traces(textposition='inside', 
                      hovertemplate=
                      "<b>%{x}</b><br>" +
                      "Prevalence: %{y:.2f}%<br>" +
                      "<extra></extra>",
                      )
    if categoryarray is not None:
        fig.update_xaxes(tickangle=tickangle, categoryorder="array", categoryarray=categoryarray)
    else:
        fig.update_xaxes(tickangle=tickangle)
    st.plotly_chart(fig, use_container_width=True)

def run(example_data):
    st.title(page_title)
    st.info(info_text, icon="ℹ️")
    
    c1, c2 = st.columns(2)
    with c1:
        # Gender
        barplot_prevalence(example_data, feature="Gender")

        # Age Group
        df_age_group = number_to_category(example_data, feature='Age', new_feature_group='Age Group')
        barplot_prevalence(df_age_group, feature="Age Group")

        # BMI Group
        df_bmi_group = number_to_category(example_data, feature="BMI", new_feature_group="BMI Group")
        group_order = ["Underweight", "Healthy Weight",  "Overweight", "Obesity"]
        barplot_prevalence(df_bmi_group, feature="BMI Group", categoryarray=group_order)

        # Heavy Drinking
        barplot_prevalence(example_data, feature="Heavy Drinking")

        # Liver Condition
        barplot_prevalence(example_data, feature="Liver Condition")

    with c2:
        # Race
        barplot_prevalence(example_data, feature="Race", tickangle=-30)

        # Sleep Duration Range
        df_sleep_group = number_to_category(example_data, feature="Sleep Duration (Hours)", new_feature_group="Sleep Duration Range")
        group_order = ["Less than 7 Hours", "7-9 Hours", "More than 9 Hours"]
        barplot_prevalence(df_sleep_group, feature="Sleep Duration Range", categoryarray=group_order)

        # Smoking
        barplot_prevalence(example_data, feature="Smoking")

        # Arthritis
        barplot_prevalence(example_data, feature="Arthritis")

        # Parental Osteoporosis
        barplot_prevalence(example_data, feature="Parental Osteoporosis")
