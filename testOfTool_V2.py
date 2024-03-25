import httpx
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import json
import asyncpg
import os
import asyncio
import datetime

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StructuredDataset
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from aif360.explainers import MetricTextExplainer
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.detectors.mdss.ScoringFunctions import Bernoulli
from aif360.detectors.mdss.MDSS import MDSS
from aif360.detectors.mdss.generator import get_random_subset



async def create_db_connection():
    global db_conn
    db_user = os.getenv('POSTGRES_USER', 'default_user')
    db_password = os.getenv('POSTGRES_PASSWORD', 'default_password')
    db_name = os.getenv('POSTGRES_DB', 'default_database')
    db_host = os.getenv('POSTGRES_HOST', '128.2.205.115')
    db_port = os.getenv('POSTGRES_PORT', 5432)
    db_conn = await asyncpg.create_pool(user=db_user, password=db_password, database=db_name, host=db_host, port=db_port)

async def close_db_connection():
    if db_conn:
        await db_conn.close()

async def fetch_data(query):
    async with db_conn.acquire() as connection:
        return await connection.fetch(query)

# Function to split genres and create separate rows
def split_genres(df):
    split_df = df['genres'].str.split(',').apply(lambda x: [genre.strip("[]") for genre in x])
    stacked_df = split_df.explode().reset_index(drop=True).rename('genre')
    return df.drop('genres', axis=1).join(stacked_df)


async def bias_impact(dataset_df):

    # Define privileged and unprivileged gender groups
    privileged_gender_male = [{'gender': 1}]  # Assuming males are the privileged group
    unprivileged_gender_female = [{'gender': 0}]  # Assuming females are the unprivileged group

    # Define privileged and unprivileged age groups
    privileged_age_groups_young = [{'age_group': 1}]
    unprivileged_age_groups_old = [{'age_group': 0}]

    # Define a function to create datasets
    def create_dataset(data, protected_attribute):
        return BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=data,
            label_names=[protected_attribute],
            protected_attribute_names=[protected_attribute]
        )
        
        
    # Create datasets for gender and age groups
    # Create datasets for gender and age groups
    dataset_gender = create_dataset(dataset_df, 'gender')
    dataset_age_group = create_dataset(dataset_df, 'age_group')
    
    # Define a function to compute fairness metrics
    def compute_fairness_metrics(dataset, privileged_groups, unprivileged_groups):
        metric = BinaryLabelDatasetMetric(dataset, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
        return metric.mean_difference()

# Define a function to apply reweighing for fairness mitigation
    def apply_reweighing(dataset, privileged_groups, unprivileged_groups):
        reweighing = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        transformed_dataset = reweighing.fit_transform(dataset)
        transformed_metric = BinaryLabelDatasetMetric(transformed_dataset, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
        return transformed_metric.mean_difference()

    # Compute fairness metrics for gender
    print("Gender fairness metrics:")
    print(compute_fairness_metrics(dataset_gender, privileged_groups=privileged_gender_male, unprivileged_groups=unprivileged_gender_female))

    # Compute fairness metrics for age groups
    print("Age group fairness metrics:")
    print(compute_fairness_metrics(dataset_age_group, privileged_groups=privileged_age_groups_young, unprivileged_groups=unprivileged_age_groups_old))

    # Apply reweighing for fairness mitigation for gender
    print("Fairness metrics after reweighing for gender:")
    print(apply_reweighing(dataset_gender, privileged_groups=privileged_gender_male, unprivileged_groups= unprivileged_gender_female))

    # Apply reweighing for fairness mitigation for age groups
    print("Fairness metrics after reweighing for age groups:")
    print(apply_reweighing(dataset_age_group, privileged_groups=privileged_age_groups_young, unprivileged_groups=unprivileged_age_groups_old))


   
async def main():
    await create_db_connection()
    
    # Fetch data from the database

    # Get current date and calculate 5 days ago and 10 days ago
    current_date = datetime.datetime.now().date()
    date_7_days_ago = current_date - datetime.timedelta(days=7)
    date_14_days_ago = current_date - datetime.timedelta(days=14)

    # Format dates for SQL query
    date_7_days_ago_str = date_7_days_ago.strftime("%Y-%m-%d %H:%M:%S")
    date_14_days_ago_str = date_14_days_ago.strftime("%Y-%m-%d %H:%M:%S")
    
    data_query = f"""
    WITH REQUESTS_TABLE AS (
    SELECT
    ROW_NUMBER() OVER() AS requestid, 
    userid, 
    time,
    unnest(string_to_array(replace(result, ' result: ', ''), ', ')) as movieid
    FROM request
    WHERE status = 200
    AND time::date BETWEEN '2024-02-19' AND '2024-02-25'
    ),
    USERS_WATCH AS (
    SELECT userid, movieid,
    time,
    watchtime
    FROM watch
    WHERE time::date BETWEEN '2024-02-19' AND '2024-02-28'
    ),
    JOIN_TABLE AS (
    SELECT 
    RT.requestid,
    RT.userid,
    DATE(RT.time) AS date,
    RT.movieid,
    UW.watchtime
    FROM REQUESTS_TABLE AS RT
    LEFT JOIN USERS_WATCH AS UW
    ON RT.userid = UW.userid 
    AND RT.movieid = UW.movieid
    AND AGE(UW.time, RT.time) <= INTERVAL '3 days'
    AND UW.time > RT.time
    )
    SELECT jt.requestid, jt.userid, jt.date, jt.movieid, jt.watchtime, u.age, u.gender, u.occupation
    FROM JOIN_TABLE jt 
    JOIN users u
    ON jt.userid = u.user_id
    ORDER BY date
    """

    # Execute the combined query and fetch the result
    # Assuming you have a function to execute SQL queries and fetch results
    data_result = await fetch_data(data_query)
    #print(data_result)
    # Convert the result to a DataFrame
    data_df = pd.DataFrame(data_result, columns=['requestID', 'userid', 'date','movieName','watchtime', 'age', 'gender', 'occupation'])
    #print(data_df.head())
    #print(data_df.shape)
    data_df['gender'] = data_df['gender'].apply(lambda x: 1 if x == 'M' else 0)
    data_df.fillna(0, inplace=True)
    #print(data_df['watchtime'].unique())

    # Create a LabelEncoder object
    label_encoder = LabelEncoder()

    # Encode the 'movieName' column
    data_df['movieName'] = label_encoder.fit_transform(data_df['movieName'])
    
    # Map age ranges to age groups
    def map_age_to_group(age):
        if age < 30:
            return 'young'
        elif age >= 30 and age < 60:
            return 'middle-aged'
        else:
            return 'senior'

    def watch_movie(watchtime):
        if watchtime<30:
            return 0
        else: return 1
        
    def watchedOrNo(prob):
        if prob<0.365:
            return 0
        else: return 1
        
    # Function to generate random probabilities within the specified ranges
    def generate_probabilities(gender):
        if gender == 1:
            return np.random.uniform(0.25, 0.99)
        elif gender == 0:
            return np.random.uniform(0.01, 0.38)


        
    data_df['watchtime_YN'] = data_df['watchtime'].apply(watch_movie)
    

    # Create the watch_predicted column based on the generated probabilities
    data_df['predicted_probability'] = data_df['gender'].apply(generate_probabilities)
    data_df['predicted_watch'] = data_df['predicted_probability'].apply(watchedOrNo)
    
    
    
    print(data_df.head())
    data_df.to_csv('usersWatch.csv', index=False)
    # Apply mapping to create 'age_group' column
    #data1_split['age_group'] = data1_split['age'].apply(map_age_to_group)
    # Change rating values of 4 and 5 to 1, and values of 1, 2, and 3 to 0
    #data1_split['age_group'] = data1_split['age_group'].apply(lambda x: 1 if x in ["young","middle-aged"] else 0)

    #await bias_impact(data_df)
    print(f"number of (instances, attributes) in the dataset = {data_df.shape}")
    print(f"stats of true likes (1=yes, 0=no) = {data_df['watchtime_YN'].value_counts()}")
    print(f"stats of predicted likes (1 = yes, 0 = no) = {data_df.size}")
    await close_db_connection()


 
asyncio.run(main())