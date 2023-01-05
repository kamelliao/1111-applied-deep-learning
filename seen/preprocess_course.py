from pathlib import Path
import yaml

from bs4 import BeautifulSoup
from ckiptagger import WS
import pandas as pd


if __name__ == '__main__':
    cfg = yaml.safe_load(open('config.yml', 'r'))

    COURSE_FILE = Path(cfg['data_path'], 'courses.csv')
    TOKENIZER_PATH = 'seen/resources/data'
    TOKENIZED_COURSE_FILE = 'seen/resources/courses_tokenized.json'
    TOKENIZED_CLEAN_COURSE_FILE = 'seen/resources/courses_tokenized_clean.json'

    PUNC = "！？｡。＂＃＄％＆＇（）＊＋，，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏. "
    COURSE_TEXT_FIELDS = [
        'course_name',
        'teacher_intro',
        'description',
        'will_learn',
        'required_tools',
        'recommended_background',
        'target_group'
    ]

    courses = pd.read_csv(COURSE_FILE)

    # html -> text
    courses['description'] = courses.description.apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
    courses = courses.fillna(' ')

    # tokenization
    tokenizer = WS(TOKENIZER_PATH, disable_cuda=True)

    for field in COURSE_TEXT_FIELDS:
        print(f'Tokenizing "{field}"...')
        courses[field] = tokenizer(courses[field])

    courses.to_json(TOKENIZED_COURSE_FILE, orient='records', lines=True, force_ascii=False)
    print(f'Tokenized course list save to "{TOKENIZED_COURSE_FILE}"')

    # cleaning
    courses = pd.read_json(TOKENIZED_COURSE_FILE, lines=True)
    for field in COURSE_TEXT_FIELDS:
        courses[field] = courses[field].apply(
            lambda tokens: ' '.join([t.strip() for t in tokens 
                if (
                    t not in PUNC
                    and t.isalnum()
                    and not t.isnumeric()
                )
            ])
        )
    courses.to_json(TOKENIZED_CLEAN_COURSE_FILE, orient='records', lines=True, force_ascii=False)
    print(f'Tokenized clean course list save to "{TOKENIZED_CLEAN_COURSE_FILE}"')