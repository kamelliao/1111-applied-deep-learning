# ADL Final Project, 2022 Autumn
## Task description

## Instructions
### Environment
```bash
pip install -r requirements.txt
```

## Seen domain(courses)
To tokenize course description, run
```bash
python3 seen/preprocess_course.py
```

To one hot encode each user, run
```bash
python3 seen/preprocess_train.py
```

Now, we are ready to obtain prediction on test data, run
```bash
# output file's name: preds_test_seen_cf.csv
python3 seen/main.py --test
```

## Seen domain(subgroup)
First you need to predict the seen domain courses by above steps.
Then, using the output csv to predict seen domain subgroup.
```bash
# output file's name: preds_sub_cf.csv
python3 seen/course2subgroup.py
```

## Unseen domain(courses)
To label-encode each categorical columns in user data and course data, run
```bash
python3 unseen/vocab.py
python3 unseen/preprocess.py
python3 unseen/vocab_multihot.py
```
For model training and prediction, run
```bash
# output file's name: preds.csv
python3 unseen/main.py
```
## Unseen domain(Subgroup)
First you need to predict the unseen domain courses by above steps.

Then, using the pred.csv to predict unseen domain subgroup.
```shell
# output file's name: unseen_group_pred.csv
python unseen/course2subgroup.py
```
