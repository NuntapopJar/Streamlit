# Import necessary libraries
import streamlit as st
import pandas as pd
from scipy.spatial import distance
from datetime import timedelta
from sklearn.cluster import KMeans
import numpy as np

# Function to calculate Euclidean distance between LAB color values
def calculate_color_distance(row1: pd.Series, row2: pd.Series) -> float:
    lab1 = row1[['LAB1', 'LAB2', 'LAB3', 'LAB4']].fillna(0).to_numpy()
    lab2 = row2[['LAB1', 'LAB2', 'LAB3', 'LAB4']].fillna(0).to_numpy()
    return distance.euclidean(lab1, lab2)

# Enhanced function to process the scheduling with improved LAB color similarity prioritization
def process_scheduling(df: pd.DataFrame) -> dict:
    try:
        # Convert datetime columns to datetime type for sorting and calculations
        df['due_datetime'] = pd.to_datetime(df['due_datetime'], errors='coerce')
        df['job_number'] = df['job_number'].astype(str)  # Ensure job_number is treated as text

        # Check for necessary columns
        required_columns = {'Machine_Name', 'due_datetime', 'planned_run_time', 'planned_setup_time', 'LAB1', 'LAB2', 'LAB3', 'LAB4'}
        if not required_columns.issubset(df.columns):
            raise ValueError("Input DataFrame is missing one or more required columns.")

        # Remove any entries where Machine_Name is 'Applied filters'
        df = df[df['Machine_Name'] != 'Applied filters']

        # Prepare a dictionary to store schedule for each machine
        schedule = {}

        # Group by Machine
        for machine, group in df.groupby('Machine_Name'):
            # Sort by due_datetime (Rank 1)
            group = group.sort_values(by='due_datetime')

            # Cluster jobs by LAB color similarity to group similar colors (Rank 2)
            lab_features = group[['LAB1', 'LAB2', 'LAB3', 'LAB4']].fillna(0).to_numpy()
            n_clusters = min(len(group), 5)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                group['color_cluster'] = kmeans.fit_predict(lab_features)
            else:
                group['color_cluster'] = 0  # All in one cluster if only one job

            # Sort by clusters first, then by due date
            sorted_group = group.sort_values(by=['color_cluster', 'due_datetime'])

            # Further sort by Size and Plate_No (Rank 3) within each cluster
            sorted_group = sorted_group.sort_values(by=['color_cluster', 'due_datetime', 'Size', 'Plate_No'])

            # Adjust start times based on planned setup and run times
            sorted_group = sorted_group.reset_index(drop=True)
            sorted_group['adjusted_start_datetime'] = pd.Timestamp('now')  # Initial placeholder for start time

            for i in range(len(sorted_group)):
                if i == 0:
                    sorted_group.loc[i, 'adjusted_start_datetime'] = pd.Timestamp('now')  # Earliest possible time
                else:
                    prev_end_time = sorted_group.loc[i-1, 'adjusted_start_datetime'] + timedelta(
                        minutes=int(sorted_group.loc[i-1, 'planned_run_time']) + int(sorted_group.loc[i-1, 'planned_setup_time'])
                    )
                    sorted_group.loc[i, 'adjusted_start_datetime'] = max(prev_end_time, pd.Timestamp('now'))

            # Drop the original start_datetime column to avoid confusion
            sorted_group = sorted_group.drop(columns=['start_datetime'], errors='ignore')

            # Store the schedule for the current machine
            schedule[machine] = sorted_group

        return schedule
    
    except Exception as e:
        st.error(f"An error occurred while processing the schedule: {e}")
        return {}

# Function to convert DataFrame to CSV and return it as a downloadable file
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

# Streamlit app
def main():
    st.title('Job Scheduling App')

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            
            # Process the schedule
            schedule = process_scheduling(df)

            # Display the schedules and provide download options
            for machine_name, machine_schedule in schedule.items():
                if machine_name == 'Applied filters':
                    continue  # Skip the machine named 'Applied filters'
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
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    main()
