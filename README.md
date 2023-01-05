# ADL Final Project, 2022 Autumn
## Task description

## Instructions
### Environment
```bash
pip install -r requirements.txt
```

### Seen domain
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

### Unseen domain
To label-encode each categorical columns in user data and course data, run
```bash
python3 unseen/vocab.py
python3 unseen/preprocess.py
```
For model training and prediction, run
```bash
python3 unseen/main.py
```