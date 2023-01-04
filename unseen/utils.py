from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# from vocab import UNK_TOKEN

def load_datasets():
    dataset_dir = Path('hahow/data')
    datasets = {data_path.stem: pd.read_csv(data_path) for data_path in dataset_dir.iterdir()}

    return datasets


def train2feat():
    datasets = load_datasets()
    all_users = datasets['users'].user_id
    all_courses = datasets['courses'].course_id
    
    train_data = datasets['train'].copy()
    train_data['course_id'] = train_data['course_id'].str.split()
    train_data_long = train_data.explode('course_id', ignore_index=True)

    breakpoint()
    # users = pd.get_dummies(train_data_long['user_id'])
    # courses = pd.get_dummies(train_data_long['course_id'].astype(pd.CategoricalDtype(categories=all_courses)))
    # # feats = pd.concat([users, courses], axis=1)

    # users.to_csv('users_onehot.csv', index=False)
    # courses.to_csv('course_onehot.csv', index=False