import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


if __name__ == '__main__':
    train = pd.read_csv('hahow/data/train.csv')
    courses = pd.read_csv('hahow/data/courses.csv')

    lbl = MultiLabelBinarizer()
    lbl.fit(courses.course_id.apply(lambda x: [x]))   

    train['course_id'] = train.course_id.str.split()
    train['course_id'] = lbl.transform(train['course_id']).tolist()
    train = pd.concat([train['user_id'], train['course_id'].apply(pd.Series).add_prefix('course_id_')], axis=1)

    train.to_csv('resources/train_onehot.csv', index=False)