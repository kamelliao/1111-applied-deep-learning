# ADL Final Project, 2022 Autumn
## Task description

## Instructions
### Environment
```bash
pip install -r requirements.txt
```

### Seen domain
```
cd seen
```
To tokenize course description, run
```bash
python3 preprocess_course.py
```

To one hot encode each user, run
```bash
python3 preprocess_train.py
```

Now, we are ready to obtain prediction on test data, run
```bash
python3 main.py --test
```

### Unseen domain
```bash
cd unseen
```

To label-encode each categorical columns in user data and course data, run
```bash
python3 vocab.py
python3 preprocess.py
```
For model training and prediction, run
```bash
python3 main.py
```