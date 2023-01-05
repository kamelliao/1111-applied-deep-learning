import yaml

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from vocab import UNK_TOKEN, USER_FEATS, COURSE_FEATS
from utils import load_datasets


def userfeat_transform(col: pd.Series):
    if col.name not in USER_FEATS:
        return col

    lbl = LabelEncoder()
    lbl.classes_ = np.load(f'unseen/resources/user/{col.name}.npy')
    col = col.str.split(',').apply(lambda x: lbl.transform(x))
    return col


def coursefeat_transform(col: pd.Series):
    if col.name not in COURSE_FEATS:
        return col

    lbl = LabelEncoder()
    lbl.classes_ = np.load(f'unseen/resources/course/{col.name}.npy')
    col = col.str.split(',').apply(lambda x: lbl.transform(x))
    return col


if __name__ == '__main__':
    cfg = yaml.safe_load(open('config.yml', 'r'))

    datasets = load_datasets(cfg['data_path'])
    users = datasets['users'].copy().fillna(UNK_TOKEN)
    users = users.apply(userfeat_transform)
    users.to_json('users.json', orient='records', lines=True)

    courses = datasets['courses'].copy().fillna(UNK_TOKEN)
    courses = courses.apply(coursefeat_transform)
    courses = courses[['course_id'] + COURSE_FEATS]
    courses.to_json('courses.json', orient='records', lines=True)