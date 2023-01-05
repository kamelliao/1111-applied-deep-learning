from collections import Counter

import pandas as pd

from main import load_datasets


def course2subgroups(course_ids: pd.Series):
    subgroup = [courses[cid]['sub_groups'].split(',') for cid in course_ids if isinstance(courses[cid]['sub_groups'], str)]
    counter = Counter()
    for cid in course_ids[:100]:
        if isinstance(courses[cid]['sub_groups'], str):
            subgroup = courses[cid]['sub_groups'].split(',')
            counter.update(subgroup)

    subgroup = counter.most_common(50)
    subgroup = ' '.join([str(subgroups[sname]) for sname, _ in subgroup])
    return subgroup


if __name__ == '__main__':
    datasets = load_datasets()
    courses = datasets['courses'].copy()
    courses = {r['course_id']: r.to_dict() for _, r in courses.iterrows()}
    subgroups = datasets['subgroups'].copy()
    subgroups = {r['subgroup_name']: r['subgroup_id'] for _, r in subgroups.iterrows()}

    preds = pd.read_csv('preds_test_seen_cf.csv')
    preds['course_id'] = preds.course_id.str.split()
    preds['subgroup'] = preds.course_id.apply(course2subgroups)
    
    preds[['user_id', 'subgroup']].to_csv('preds_sub_cf.csv', index=False)