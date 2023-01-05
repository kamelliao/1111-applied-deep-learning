import yaml
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


if __name__ == '__main__':
    cfg = yaml.safe_load(open('config.yml', 'r'))

    train = pd.read_csv(Path(cfg['data_path'], 'train.csv'))
    courses = pd.read_csv(Path(cfg['data_path'], 'courses.csv'))

    lbl = MultiLabelBinarizer()
    lbl.fit(courses.course_id.apply(lambda x: [x]))   

    train['course_id'] = train.course_id.str.split()
    train['course_id'] = lbl.transform(train['course_id']).tolist()
    train = pd.concat([train['user_id'], train['course_id'].apply(pd.Series).add_prefix('course_id_')], axis=1)

    train.to_csv('seen/resources/train_onehot.csv', index=False)