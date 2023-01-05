import os
import yaml

import joblib
from pandas import Series
from sklearn.preprocessing import MultiLabelBinarizer

from utils import load_datasets


UNK_TOKEN = 'UNK'
USER_FEATS = ['gender', 'occupation_titles', 'interests', 'recreation_names']
COURSE_FEATS = ['course_id', 'teacher_id', 'groups', 'sub_groups', 'topics']
RSRC_PATH = 'unseen/resources/'


def build_vocab(field: Series, output_path: str):
    labels = set()
    for _, items in field.items():
        # to prevent update on 'nan'
        if isinstance(items, list):
            labels.update(items)
        else:
            labels.add(items)
    
    lbl = MultiLabelBinarizer()
    lbl.fit([list(labels)])  # note: should be list of lists
    joblib.dump(lbl, output_path)

    print(f'Vocab successfully saved to {output_path}')


if __name__ == '__main__':
    cfg = yaml.safe_load(open('config.yml', 'r'))
    datasets = load_datasets(cfg['data_path'])

    os.makedirs('unseen/resources/user', exist_ok=True)
    os.makedirs('unseen/resources/course', exist_ok=True)

    # users
    users = datasets['users'].fillna(UNK_TOKEN)
    for prop in USER_FEATS:
        build_vocab(users[prop].str.split(','), f'unseen/resources/user/{prop}.joblib')

    # courses
    courses = datasets['courses'].fillna(UNK_TOKEN)
    for prop in COURSE_FEATS:
        build_vocab(courses[prop].str.split(','), f'unseen/resources/course/{prop}.joblib')