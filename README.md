# ADL Final Project, 2022 Autumn
## Task description

## Instructions
### Environment
```bash
pip install -r requirements.txt
```

### Seen domain(courses)
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
python3 seen/main.py --test
```

### Unseen domain(courses)
To label-encode each categorical columns in user data and course data, run
```bash
python3 unseen/vocab.py
python3 unseen/preprocess.py
```
For model training and prediction, run
```bash
python3 unseen/main.py
```
### Unseen domain(Subgroup)
First you need to predict the unseen domain courses by above steps.

Then, using the pred.csv to predict unseen domain subgroup.
```shell
python unseen/course2subgroup.py

# If you wnat to Specify input & output file path
python unseen/course2subgroup.py --unseen_course_pred {1} --output_file {2}
# {1}: the output file of unseen domain courses, default="preds.csv"
# {2}: where you want to store the output file, default="unseen_group_pred.csv"
```