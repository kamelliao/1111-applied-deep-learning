from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Plus


COURSE_FEATS = [
    'course_name',
    'teacher_intro',
    'description',
    'will_learn',
    'required_tools',
    'recommended_background',
    'target_group'
]


def load_datasets():
    dataset_dir = Path('hahow/data')
    datasets = {data_path.stem: pd.read_csv(data_path) for data_path in dataset_dir.iterdir()}

    return datasets


def get_content_sim(method='bm25'):
    courses_df = pd.read_json('resources/courses_tokenized_clean.json', lines=True)
    courses_df.sort_values('course_id', inplace=True)  # to align with MultiLabelBinarizer's order

    # build corpus
    corpus = []
    for _, row in courses_df[COURSE_FEATS].iterrows():
        corpus.append(' '.join(row))
    
    # build index
    if method == 'tfidf':
        model = TfidfVectorizer(max_df=0.5, stop_words=Path('resources/stopwords.txt').read_text().split('\n'))
        corpus_tfidf = model.fit_transform(corpus)
        content_sim = cosine_similarity(corpus_tfidf, corpus_tfidf)
    elif method == 'bm25':
        corpus_tokenized = [doc.split(' ') for doc in corpus]
        model = BM25Plus(corpus_tokenized, k1=0.3, b=0.1)
        content_sim = np.array([model.get_scores(doc) for doc in tqdm(corpus_tokenized)])
    
    np.fill_diagonal(content_sim, 0)
    return content_sim


def inference(split='val_seen'):
    valid = pd.merge(datasets[split], train, on='user_id')
    valid = valid.drop(columns=['course_id'])
    user_item = np.dot(valid[valid.columns[1:]].values, iisim)
    # np.save('resources/user_item.npy', user_item)

    preds = []
    for user in user_item:
        rcm_courses = (-user).argsort()[:]
        rcm_courses = ' '.join(lbl.classes_[rcm_courses])
        preds.append(rcm_courses)
    
    pd.DataFrame({'user_id': valid.user_id, 'course_id': preds}).to_csv(f'preds_{split}_cf.csv', index=False)
    print(f'Prediction results saved to "preds_{split}_cf.csv"')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--metric', type=str, default='dot')
    parser.add_argument('-c', '--content', type=str, default='bm25')
    parser.add_argument('-l', '--lam', type=float, default=0.05)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    datasets = load_datasets()
    courses = datasets['courses'].copy()
    lbl = MultiLabelBinarizer()
    lbl.fit(courses.course_id.apply(lambda x: [x]))   

    # co-purchase frequency
    train = pd.read_csv('resources/train_onehot.csv')
    courses = train[train.columns[1:]].values.transpose()  # (728, 59737)

    if args.metric == 'cos':
        iisim = cosine_similarity(courses)
    elif args.metric == 'dot':
        iisim = np.dot(courses, courses.T)
    elif args.metric == 'corr':
        iisim = train[train.columns[1:]].corr()
        iisim = iisim.fillna(0).values

    # content simialrity
    if args.content == 'tfidf':
        sim_content = get_content_sim('tfidf')
    elif args.content == 'bm25':
        sim_content = get_content_sim('bm25')

    # label similarity
    # course_info = pd.read_json('ltr/resources/courses.json', lines=True)
    # course_info = course_info.sort_values('course_id')
    # course_info = np.stack(course_info.apply(lambda row: np.concatenate(row[course_info.columns[2:]]), axis=1))  # 2'teacher_id', 3'groups', 4'sub_groups', 5'topics'
    # sim_label = cosine_similarity(course_info)

    # aggregate
    iisim = np.add(iisim, args.lam*sim_content)
    # iisim = np.add(iisim, args.lam*sim_label)
    np.fill_diagonal(iisim, 0)

    # Inference
    if args.test:
        inference('test_seen')
    else:
        inference('val_seen')
