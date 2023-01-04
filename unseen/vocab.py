import os

import numpy as np
from pandas import Series
from sklearn.preprocessing import LabelEncoder

from utils import load_datasets


UNK_TOKEN = 'UNK'
USER_FEATS = ['gender', 'occupation_titles', 'interests', 'recreation_names']
COURSE_FEATS = ['teacher_id', 'groups', 'sub_groups', 'topics']


def build_vocab(field: Series, output_path: str):
    labels = set()
    for _, items in field.items():
        # to prevent update on 'nan'
        if isinstance(items, list):
            labels.update(items)
        else:
            labels.add(items)
    
    lbl = LabelEncoder()
    lbl.fit(list(labels))
    np.save(output_path, lbl.classes_)

    print(f'Vocab successfully saved to {output_path}')


if __name__ == '__main__':
    datasets = load_datasets()

    os.makedirs('./resources/user', exist_ok=True)
    os.makedirs('./resources/course', exist_ok=True)

    # users
    users = datasets['users'].fillna(UNK_TOKEN)
    for prop in USER_FEATS:
        build_vocab(users[prop].str.split(','), f'./resources/user/{prop}.npy')

    # courses
    courses = datasets['courses'].fillna(UNK_TOKEN)
    for prop in COURSE_FEATS:
        build_vocab(courses[prop].str.split(','), f'./resources/course/{prop}.npy')