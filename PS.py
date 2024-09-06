# Import necessary libraries
import streamlit as st
import pandas as pd
from scipy.spatial import distance
from datetime import timedelta
from sklearn.cluster import KMeans
import numpy as np
import os

# Path to your local image file
image_path = "SCGP_Logo_Full-Version_Isolated.png"  # Replace with your local file name

# Check if the image file exists
if os.path.exists(image_path):
    st.image(image_path, use_column_width=True)  # Display the image using the local file path
else:
    st.warning(f"Image not found at {image_path}. Please check the path or upload the image to the correct directory.")

# Function to calculate Euclidean distance between LAB color values
def calculate_color_distance(row1, row2):
    lab1 = row1[['LAB1', 'LAB2', 'LAB3', 'LAB4']].fillna(0).to_numpy()
    lab2 = row2[['LAB1', 'LAB2', 'LAB3', 'LAB4']].fillna(0).to_numpy()
    return distance.euclidean(lab1, lab2)

# Updated function to process scheduling with machine break times
def process_scheduling(df):
    # Convert datetime columns to datetime type for sorting and calculations
    df['due_datetime'] = pd.to_datetime(df['due_datetime'])
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['job_number'] = df['job_number'].astype(str)  # Ensure job_number is treated as text

    # Define break times
    break_times = [
        (timedelta(hours=11, minutes=30), timedelta(hours=12, minutes=30)),  # 11:30-12:30
        (timedelta(hours=23, minutes=0), timedelta(hours=24, minutes=0))     # 23:00-24:00
    ]

    # Function to adjust job start time based on break times
    def adjust_for_breaks(start_time, run_time_minutes):
        # Convert start time to time since the beginning of the day
        current_time_of_day = start_time.time()
        current_time_since_midnight = timedelta(hours=current_time_of_day.hour, minutes=current_time_of_day.minute)
        end_time_since_midnight = current_time_since_midnight + timedelta(minutes=run_time_minutes)

        for break_start, break_end in break_times:
            # Check if the job overlaps with the break time
            if current_time_since_midnight < break_end and end_time_since_midnight > break_start:
                # Adjust start time to the end of the break if overlapping
                start_time += (break_end - current_time_since_midnight)
                current_time_since_midnight = timedelta(hours=start_time.time().hour, minutes=start_time.time().minute)
                end_time_since_midnight = current_time_since_midnight + timedelta(minutes=run_time_minutes)

        return start_time

    # Prepare a dictionary to store schedule for each machine
    schedule = {}

    # Group by Machine
    for machine, group in df.groupby('Machine_Name'):
        # Sort by due_datetime (Rank 1)
        group = group.sort_values(by='due_datetime')

        # Cluster jobs by LAB color similarity to group similar colors (Rank 2)
        lab_features = group[['LAB1', 'LAB2', 'LAB3', 'LAB4']].fillna(0).to_numpy()
        kmeans = KMeans(n_clusters=min(len(group), 5), random_state=42).fit(lab_features)
        group['color_cluster'] = kmeans.labels_

        # Sort by clusters first, then by due date
        sorted_group = group.sort_values(by=['color_cluster', 'due_datetime'])

        # Further sort by Size and Plate_No (Rank 3) within each cluster
        sorted_group = sorted_group.sort_values(by=['color_cluster', 'due_datetime', 'Size', 'Plate_No'])

        # Identify the earliest start_datetime for this machine
        earliest_start = sorted_group['start_datetime'].min()

        # Adjust start times based on planned setup and run times
        sorted_group = sorted_group.reset_index(drop=True)
        sorted_group['adjusted_start_datetime'] = pd.Timestamp('now')  # Placeholder for start time

        for i in range(len(sorted_group)):
            if i == 0:
                # Set the start time of the first job to the earliest start time for this machine
                sorted_group.loc[i, 'adjusted_start_datetime'] = earliest_start
            else:
                prev_end_time = sorted_group.loc[i-1, 'adjusted_start_datetime'] + timedelta(
                    minutes=int(sorted_group.loc[i-1, 'planned_run_time']) + int(sorted_group.loc[i-1, 'planned_setup_time'])
                )
                sorted_group.loc[i, 'adjusted_start_datetime'] = max(prev_end_time, earliest_start)

            # Adjust the start time to account for break periods
            run_time = int(sorted_group.loc[i, 'planned_run_time']) + int(sorted_group.loc[i, 'planned_setup_time'])
            sorted_group.loc[i, 'adjusted_start_datetime'] = adjust_for_breaks(
                sorted_group.loc[i, 'adjusted_start_datetime'], run_time
            )

        # Drop the original start_datetime column to avoid confusion
        sorted_group = sorted_group.drop(columns=['start_datetime'])

        # Store the schedule for the current machine
        schedule[machine] = sorted_group

    return schedule


# Function to convert DataFrame to CSV and return it as a downloadable file
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Streamlit app
def main():
    st.title('Job Scheduling App')

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        
        # Process the schedule
        schedule = process_scheduling(df)

        # Display the schedules and provide download options
        for machine_name, machine_schedule in schedule.items():
            st.subheader(f"Schedule for Machine: {machine_name}")
            st.dataframe(machine_schedule)
            
            # Convert to CSV for download
            csv = convert_df_to_csv(machine_schedule)

            # Add download button
            st.download_button(
                label=f"Download {machine_name} Schedule as CSV",
                data=csv,
                file_name=f'{machine_name}_schedule.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
