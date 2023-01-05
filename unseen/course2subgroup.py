import csv
from pathlib import Path
from argparse import ArgumentParser, Namespace
import yaml

def main(args):
    cfg = yaml.safe_load(open('config.yml', 'r'))
    path = cfg['data_path']
    print(path)

    # 先讀預測出的course
    unseen_course = []
    with open(args.unseen_course_pred, 'r', encoding="utf-8") as f:
        rows = csv.reader(f)
        
        for row in rows:
            unseen_course.append(row)
    unseen_course = unseen_course[1:]


    for i in range(len(unseen_course)):
        unseen_course[i][1] = unseen_course[i][1].split(" ")

    # {id: pred_course}
    # unseen_course_map = {}
    # for data in unseen_course:
    #     unseen_course_map[data[0]] = data[1].split(" ")
    # print(unseen_course_map)

    # 讀所有course的資訊來獲得subgroup文字
    all_courses = []
    all_courses_map = {}
    with open(path + "/courses.csv", 'r', encoding="utf-8") as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            all_courses.append(row)
    all_courses = all_courses[1:]

    for course in all_courses:
        all_courses_map[course[0]] = course[6].split(",")

    # 讀subgroup.csv做文字到idx對照
    subgroup = []
    subgroup_map = {}
    with open(path + "/subgroups.csv", 'r', encoding="utf-8") as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            subgroup.append(row)
    for data in subgroup[1:]:
        subgroup_map[data[1]] = int(data[0])

    # 最後 將course的subgroup對應到數字 之後比較好做對照
    course_subgroup_idx = {}
    for k, v in all_courses_map.items():
        text_to_idx = []
        for text in v:
            if text != '':
                text_to_idx.append(subgroup_map[text])
        course_subgroup_idx[k] = text_to_idx
    # print(course_subgroup_idx)
        

    unseen_subgroup_pred = []
    for user in  unseen_course:                                     # 對於所有user
        pred_count = [0]*92
        for pred_courses in user[1]:                                # 的所有預測可能購買的課程
            for subgroup in course_subgroup_idx[pred_courses]:      # 課程中的所有分類，計算出現次數
                pred_count[subgroup] += 1
            
        result = []
        for i in range(len(pred_count)):
            max = 0
            max_idx = 0
            for j in range(len(pred_count)):
                if pred_count[j]>max:
                    max = pred_count[j]
                    max_idx = j
            if max > 0:
                result.append(max_idx)
                pred_count[max_idx] = 0
                max_idx = 0
                max = 0
            else:
                break
        unseen_subgroup_pred.append([user[0], result])

    # print(unseen_subgroup_pred[10000])

    with open(args.output_file, 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)

        # 寫入一列資料
        writer.writerow(['user_id', 'subgroup'])
        for data in unseen_subgroup_pred:
            s = ''
            for v in data[1]:
                s += str(v)
                s += " "
            writer.writerow([data[0], s])

# ==========================================
def parse_args() -> Namespace:
    parser = ArgumentParser()
    # data loader
    parser.add_argument("--unseen_course_pred", default="preds.csv", type=Path)

    # parser.add_argument("--courses", default="hahow/data/courses.csv", type=Path)

    # parser.add_argument("--subgroups", default="hahow/data/subgroups.csv", type=Path)

    parser.add_argument("--output_file", default="unseen_group_pred.csv", type=Path)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)