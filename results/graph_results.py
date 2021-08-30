from altair.vegalite.v4.api import value
import streamlit as st
import pandas as pd
import sys
import plotly.express as px
import numpy as np


# Function to assign a group to each data point
def find_group(episode_number, divisor):
    return int(episode_number/divisor)


# Function to get the range of values for each group
def get_range(values_list):
    retVal = None
    values_list = [int(item) for item in values_list]
    if len(values_list) == 1:
        retVal = values_list[0]
    else:
        retVal = str(min(values_list)) + " - " + str(max(values_list))
    
    return retVal


if __name__ == "__main__":
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    df['episode'] = df['episode'].astype(str)
    st.write("Original Data:")
    st.write(df)
    metric = st.sidebar.selectbox(
        "Metric",
        ("pct_complete", 
        "total_time", 
        "total_distance",
        "average_speed_kph",
        "average_displacement_error",
        "trajectory_efficiency",
        "trajectory_admissibility",
        "movement_smoothness",
        "timestep/sec"
        ))

    group_size = int(st.sidebar.selectbox(
        "Episode Group Size",
        ("1",
        "10",
        "100",
        "1000",
        "10000")))
    
    if group_size < len(df.index):
        sub_df = df[["episode", metric]]
        sub_df['group'] = sub_df.apply(lambda row : find_group(int(row['episode']), group_size), axis=1)

        sub_df = sub_df.groupby('group').agg({'episode' : list, metric : np.mean}).reset_index()
        sub_df['range'] = sub_df.apply(lambda row: get_range(row['episode']), axis=1)
        sub_df.drop(['episode'], axis=1, inplace=True)
        st.write("Aggregated Data:")
        st.write(sub_df)


        st.write("Plotting " + metric + " per episode:")
        if group_size == 1:
            fig = px.line(sub_df, x="group", y=metric)
        else:
            fig = px.line(sub_df, x="group", y=metric, hover_data=['range'])
        st.plotly_chart(fig)
    
    else:
        st.write("The group size must be less than the number of data points available.")