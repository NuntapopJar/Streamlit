# Import necessary libraries
import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
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

# Function to adjust job start time based on break times
def adjust_for_breaks(start_time, run_time_minutes, break_times):
    end_time = start_time + timedelta(minutes=run_time_minutes)
    for break_start, break_end in break_times:
        break_start_datetime = datetime.combine(start_time.date(), (datetime.min + break_start).time())
        break_end_datetime = datetime.combine(start_time.date(), (datetime.min + break_end).time())

        # If the break spans into the next day
        if break_end < break_start:
            break_end_datetime += timedelta(days=1)

        # Check for overlap
        if start_time < break_end_datetime and end_time > break_start_datetime:
            # Adjust start time to the end of the break
            start_time = break_end_datetime
            end_time = start_time + timedelta(minutes=run_time_minutes)
    return start_time

# Function to process scheduling with the updated priorities
def process_scheduling(df):
    # Convert datetime columns to datetime type for sorting and calculations
    df['due_datetime'] = pd.to_datetime(df['due_datetime'], errors='coerce')
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['job_number'] = df['job_number'].astype(str)  # Ensure job_number is treated as text

    # Define break times
    break_times = [
        (timedelta(hours=11, minutes=30), timedelta(hours=12, minutes=30)),  # 11:30-12:30
        (timedelta(hours=23, minutes=0), timedelta(hours=24, minutes=0))     # 23:00-24:00
    ]

    # Prepare a dictionary to store schedule for each machine
    schedule = {}

    # Group by Machine
    for machine, group in df.groupby('Machine_Name'):
        # Sort by due_datetime (Rank 1), Size, and Plate_No (Rank 2)
        group = group.sort_values(by=['due_datetime', 'Size', 'Plate_No'])
        group = group.reset_index(drop=True)

        # Handle missing LAB values
        # Calculate mean of LAB columns, ignoring NaNs
        lab_columns = ['LAB1', 'LAB2', 'LAB3', 'LAB4']
        lab_means = group[lab_columns].mean()
        # Fill NaNs in the means with zero (in case the entire column is NaN)
        lab_means = lab_means.fillna(0)
        # Fill NaNs in the group data with the calculated means
        group[lab_columns] = group[lab_columns].fillna(lab_means)

        # Now extract lab_features
        lab_features = group[lab_columns].to_numpy()

        # Verify that there are no NaNs in lab_features
        if np.isnan(lab_features).any():
            st.error(f"There are NaNs in LAB color columns for machine {machine} after filling missing values.")
            continue  # Skip this machine or handle appropriately

        # Perform LAB color similarity clustering (Rank 3)
        num_clusters = min(len(group), 5)
        if num_clusters < 2:
            num_clusters = 1  # KMeans requires at least one cluster
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(lab_features)
        group['color_cluster'] = kmeans.labels_

        # Sort by due_datetime, Size, Plate_No, and color_cluster
        group = group.sort_values(by=['due_datetime', 'Size', 'Plate_No', 'color_cluster'])
        group = group.reset_index(drop=True)

        # Identify the earliest start_datetime for this machine
        earliest_start = group['start_datetime'].min()

        # Initialize adjusted_start_datetime column
        group['adjusted_start_datetime'] = pd.NaT  # Placeholder for start time

        # Adjust start times based on planned setup and run times
        for i in range(len(group)):
            if i == 0:
                # Set the start time of the first job to the earliest start time for this machine
                group.loc[i, 'adjusted_start_datetime'] = earliest_start
            else:
                prev_job = group.loc[i - 1]
                prev_end_time = prev_job['adjusted_start_datetime'] + timedelta(
                    minutes=int(prev_job['planned_run_time']) + int(prev_job['planned_setup_time'])
                )
                group.loc[i, 'adjusted_start_datetime'] = max(prev_end_time, earliest_start)

            # Adjust the start time to account for break periods
            run_time = int(group.loc[i, 'planned_run_time']) + int(group.loc[i, 'planned_setup_time'])
            group.loc[i, 'adjusted_start_datetime'] = adjust_for_breaks(
                group.loc[i, 'adjusted_start_datetime'], run_time, break_times
            )

        # Drop the original start_datetime column to avoid confusion
        group = group.drop(columns=['start_datetime'])

        # Store the schedule for the current machine
        schedule[machine] = group

    return schedule

# Function to convert DataFrame to CSV and return it as a downloadable file
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Streamlit app
def main():
    st.title('Job Scheduling App')

    st.write("""
    This app schedules jobs based on the following priorities:
    1. **Due Date**
    2. **Job Size and Plate Number**
    3. **LAB Color Similarity**
    """)

    # File upload section
    st.header("Upload Job Data")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            with st.spinner('Reading the uploaded file...'):
                df = pd.read_excel(uploaded_file)
            st.success("File uploaded and read successfully!")

            # Display the uploaded data
            if st.checkbox("Show uploaded data"):
                st.subheader("Uploaded Data")
                st.dataframe(df)

            # Validate required columns
            required_columns = ['Machine_Name', 'due_datetime', 'start_datetime', 'job_number',
                                'planned_run_time', 'planned_setup_time', 'LAB1', 'LAB2', 'LAB3', 'LAB4',
                                'Size', 'Plate_No']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
                return

            # Process the schedule
            with st.spinner('Processing scheduling...'):
                schedule = process_scheduling(df)
            st.success("Scheduling completed!")

            # Display the schedules and provide download options
            st.header("Scheduling Results")
            for machine_name, machine_schedule in schedule.items():
                st.subheader(f"Schedule for Machine: {machine_name}")
                st.dataframe(machine_schedule.style.format({
                    'due_datetime': lambda t: t.strftime('%Y-%m-%d %H:%M'),
                    'adjusted_start_datetime': lambda t: t.strftime('%Y-%m-%d %H:%M') if pd.notnull(t) else ''
                }))

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
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload an Excel file to proceed.")

if __name__ == "__main__":
    main()
