from argparse import ArgumentParser
import joblib
import yaml

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from utils import load_datasets
from dataset import HahowDataset, collate_fn_multiclass, collate_fn_multiclass_inf
from model import HahowDeepFM

from vocab import UNK_TOKEN

COURSE_FEAT = ['course_id', 'sub_groups']

def get_dataset(split: str):
    all_datasets = load_datasets(cfg['data_path'])
    dataset = all_datasets[split].copy()

    users = pd.read_json('users.json', lines=True)
    # courses = pd.read_json('courses.json', lines=True)
    courses = {r['course_id']: r.to_dict() for _, r in all_datasets['courses'].fillna(UNK_TOKEN).iterrows()}
    dataset = pd.merge(dataset, users, on='user_id')

    if 'train' in split:
        dataset['course_id'] = dataset.course_id.str.split()

        lbl_subgroups: MultiLabelBinarizer = joblib.load('unseen/resources/course/sub_groups.joblib')
        dataset['label_sub_groups'] = dataset.course_id.apply(
            lambda clist: lbl_subgroups.transform([courses[cid]['sub_groups'].split(',') for cid in clist]).sum(axis=0)
        )


    lbl: MultiLabelBinarizer = joblib.load(f'unseen/resources/course/course_id.joblib')
    dataset[f'label_course_id'] = lbl.transform(dataset['course_id']).tolist()

    return dataset


if __name__ == '__main__':
    cfg = yaml.safe_load(open('config.yml', 'r'))

    parser = ArgumentParser()
    parser.add_argument('-t', '--test_file', type=str, default='test_unseen', help='[ val_seen | val_unseen | test_seen | test_unseen ]')
    args = parser.parse_args()

    torch.manual_seed(42)

    train_set = get_dataset('train')
    valid_set = get_dataset(args.test_file)

    # set up model
    ebc = EmbeddingBagCollection(tables=[
        EmbeddingBagConfig(name='u1', embedding_dim=2, num_embeddings=4, feature_names=['gender']),
        EmbeddingBagConfig(name='u2', embedding_dim=2, num_embeddings=96, feature_names=['interests'], pooling=PoolingType.MEAN),
        EmbeddingBagConfig(name='u3', embedding_dim=2, num_embeddings=21, feature_names=['occupation_titles']),
        EmbeddingBagConfig(name='u4', embedding_dim=2, num_embeddings=32, feature_names=['recreation_names'], pooling=PoolingType.MEAN),
    ])


    model = HahowDeepFM(
        embedding_bag_collection=ebc,  # sparse features
        deep_fm_dimension=20,  # dnn output dimension
    )

    train_dataset = HahowDataset(train_set)
    valid_dataset = HahowDataset(valid_set)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn_multiclass)
    valid_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=False, collate_fn=collate_fn_multiclass_inf)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    loss_fn = torch.nn.BCELoss()

    # training
    model.to('cuda')
    model.train()
    for e in range(10):
        train_loss = 0
        for _, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            labels_courseid, labels_subgroup = batch.pop('labels_courseid'), batch.pop('labels_subgroup')

            pred_courseid, pred_subgroup = model(**batch)
            loss = loss_fn(torch.cat([pred_courseid, pred_subgroup], dim=1), torch.cat([labels_courseid, labels_subgroup], dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Epoch {(e+1):<2} | loss {(train_loss/len(train_loader)):.5f}')

    # evaluation
    results = []
    model.eval()
    with torch.no_grad():

        for _, batch in enumerate(tqdm(valid_loader)):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            # labels = batch.pop('labels')
            # labels_courseid, labels_subgroup = batch.pop('labels_course_id'), batch.pop('labels_sub_groups')

            pred, _ = model(**batch)
            top50 = pred.argsort(axis=-1, descending=True)#[:, :50]
            results.extend(top50.tolist())

    # write prediction results
    lbl: MultiLabelBinarizer = joblib.load('unseen/resources/course/course_id.joblib')
    lbl_map = {cid: cname for cid, cname in enumerate(lbl.classes_)}

    results = [[lbl_map[r] for r in rl] for rl in results]
    preds = pd.DataFrame({'user_id': valid_set.user_id.unique().tolist(), 'course_id': results})
    preds['course_id'] = preds.course_id.apply(lambda x: ' '.join(x))
    preds.to_csv('preds.csv', index=False)