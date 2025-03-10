import os
from functools import partial
import httpx

import random
import copy
import json
import re
from collections import Counter

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

import numpy as np
import cv2
import math


IOF_THR = 0.55
RBOX_NORM = 100.0
OBJ_NUM_THR = 20
RBOX_START = '<rbox>'
RBOX_END = '</rbox>'


# 定义问题列表
# 1) task1 Object Detection  [detection]
# Input: text(category)  Output: region
Object_Detection_QUESTIONS = [
    "Can you locate all the <category> in the image?",
    "Could you help me find all the <category> in the image? Please provide their locations.",
    "Detect all the <category> in the image and output their locations.",
    "Detect all the <category> and output their locations.",
    "Provide the coordinates of all <category> in the image.",
    "Can you find and mark the positions of all the <category> in the given image?",
    "Please detect all the <category> in the image and output their locations.",
    "Locate and list the positions of all <category> that appear in the image.",
    "Identify and provide the coordinates of all <category> in the image.",
    "Identify all the <category> and mark their locations.",
    "I need you to detect and locate all <category> present in the image.",
    "Detect the locations of all <category> objects in the provided image.",
    "Please locate all the <category> in the given image."
    ]

NEG_ANSWER1 = "I'm sorry, I cannot answer as the given image does not contain any given objects."


def poly2obb_np(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_np_oc(polys)
    elif version == 'le135':
        results = poly2obb_np_le135(polys)
    elif version == 'le90':
        results = poly2obb_np_le90(polys)
    else:
        raise NotImplementedError
    return results

def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')


def poly2obb_np_oc(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    # if w < 2 or h < 2:
    #     return
    while not 0 < a <= 90:
        if a == -90:
            a += 180
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 < a <= np.pi / 2
    return x, y, w, h, a


def poly2obb_np_le135(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)
    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))
    if edge1 < 2 or edge2 < 2:
        return
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))
    angle = norm_angle(angle, 'le135')
    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return x_ctr, y_ctr, width, height, angle


def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a

Fair_special_class = {
    'airplane': ['A220', 'A321', 'A330', 'A350', 'ARJ21', 'Boeing737', 'Boeing747', 'Boeing777',
                 'Boeing787', 'C919', 'other-airplane'],
    'ship':['Passenger_Ship', 'Engineering_Ship', 'Liquid_Cargo_Ship', 'Dry_Cargo_Ship', 'Warship', 'other-ship'],
    'boat':['Motorboat', 'Fishing_Boat', 'Tugboat']    
}

def get_primary_category(cat_name):
    for primary_cat, secondary_cats in Fair_special_class.items():
        if cat_name in secondary_cats:
            return primary_cat
    return cat_name

def generate_Object_Detection_QA(image_name, objects, obj_categories, category_to_id,
                                 create_neg_sample=True):

    img_w = 512.
    img_h = 512.
    prefix = "[detection]"
    # 统计当前图像中包含的类别
    present_categories = list(set([obj['category_id'] for obj in objects]))

    rboxs_per_category_list = []
    for cat_id in present_categories:
        rboxs_per_category_list.append([])

    # 单个/多个目标的grounding
    for obj in objects:
        obj_id = obj['object_id']
        cat_id = obj['category_id']
        cat_name = obj_categories[cat_id]

        rbox = obj['rbox']
        iof = obj['box_iof']
        if iof < IOF_THR:
            continue
        cx,cy,w,h,a = poly2obb_np(np.array(rbox, dtype=np.float32))
        # normalize
        cx_, cy_, w_, h_ = (round(coord / img_w * RBOX_NORM, 2) for coord in (cx, cy, w, h))
        a_degrees = math.degrees(a)
        # rbox_str = "{<%d><%d><%d><%d>|<%d>}" % (cx_, cy_, w_, h_ , a_degrees)
        rbox_str = "{<%.2f><%.2f><%.2f><%.2f>|<%d>}" % (cx_, cy_, w_, h_ , a_degrees)
        category_index = present_categories.index(cat_id)
        rboxs_per_category_list[category_index].append(rbox_str)

    # Create question answers
    questions_answers = []
    for cat_id in present_categories:
        cat_index = present_categories.index(cat_id)
        rbox_list = rboxs_per_category_list[cat_index]
        cat_num = len(rbox_list)
        answer_end = ""
        if cat_num > OBJ_NUM_THR:
            rbox_list = rbox_list[:OBJ_NUM_THR]  # 注意设置最大数量限制, 数量过多则不输出全部坐标
            answer_end = ", and due to the context length, the remaining objects are not listed."

        cat_name = obj_categories[cat_id]
        cat_name = get_primary_category(cat_name)  # Fair1m合并类别

        cat_name_str = cat_name.replace('-', '_').lower()  # 替换'-'为'_'

        answer_str = RBOX_START + '(' + ", ".join(rbox_list) + ')' + RBOX_END
        if cat_num == 1:
            pre_answer = f"There is {cat_num} {cat_name_str} in the image:"
        elif cat_num == 0:
            pre_answer = NEG_ANSWER1
            answer_str = ""
        else:
            pre_answer = f"There are {cat_num} {cat_name_str}s in the image:"
        answer = pre_answer + " " + answer_str + answer_end
        question_template = random.choice(Object_Detection_QUESTIONS)
        question_with_cat = prefix + question_template.replace('<category>', cat_name_str + 's')
        questions_answers.append((image_name, question_with_cat, answer))
    
    ## 构建负样本
    if create_neg_sample:
        absent_categories = [cat_id for cat_id in range(len(obj_categories)) if cat_id not in present_categories]
        # random select 1-2 classes
        # selected_absent_categories = random.sample(absent_categories, k=random.randint(1,2))
        selected_absent_categories = random.sample(absent_categories, 1)
        for cat_id in selected_absent_categories:
            cat_name = obj_categories[cat_id]
            cat_name = get_primary_category(cat_name)  # Fair1m合并类别
            cat_name_str = cat_name.replace('-', '_').lower()
            question_template = random.choice(Object_Detection_QUESTIONS)
            neg_question_with_cat = prefix + question_template.replace('<category>', cat_name_str + 's')
            neg_answer = NEG_ANSWER1  # negaive answer
            questions_answers.append((image_name, neg_question_with_cat, neg_answer))

    return questions_answers

def Process_Dataset(anno_path, obj_categories, category_to_id, type='train'):

    question_answers = []
    if type == 'train':
        data_path = os.path.join(anno_path, 'train/annfiles/')
    elif type == 'test':
        data_path = os.path.join(anno_path, 'val/annfiles/')

    for filename in os.listdir(data_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_path, filename)
            image_name = os.path.splitext(filename)[0] + '.png'
            objects = []
            with open(filepath, 'r') as file:
                for obj_id, line in enumerate(file):
                    parts = line.strip().split()
                    if len(parts) > 8:
                        rbox = list(map(float, parts[:8]))
                        category = parts[8]
                        difficulty = int(parts[9])
                        if difficulty == 0:
                            category_id = category_to_id[category]
                            objects.append({
                                'object_id': obj_id,
                                'category_id': category_id,
                                'rbox': rbox,
                                'box_iof': 1.0
                            })
            if objects:
                qa_pairs = generate_Object_Detection_QA(image_name, objects, obj_categories, category_to_id,
                                                        create_neg_sample=True)
                
                question_answers = question_answers + qa_pairs

    return question_answers

######
## 统计类别词表
ori_path_DOTA = "/media/dell/data1/ljw/data/DOTA-v2.0/train/labelTxt-v2.0/DOTA-v2.0_train/"
ori_path_Fair = "/media/dell/data1/ljw/data/FAIR1M1.0/fair1m_dota-format/train/labelTxt/"

# 用于存储所有类别的集合
obj_categories_DOTA = set()
obj_categories_Fair = set()

# 遍历目录中的所有文件，收集类别信息
for filename in os.listdir(ori_path_DOTA):
    if filename.endswith('.txt'):
        filepath = os.path.join(ori_path_DOTA, filename)
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) > 8:
                    category = parts[8]
                    difficulty = int(parts[9])
                    if difficulty == 0:
                        obj_categories_DOTA.add(category)

for filename in os.listdir(ori_path_Fair):
    if filename.endswith('.txt'):
        filepath = os.path.join(ori_path_Fair, filename)
        with open(filepath, 'r') as file:
            for i, line in enumerate(file):
                if i < 2:  # 跳过前两行
                    continue
                parts = line.strip().split()
                if len(parts) > 8:
                    category = parts[8]
                    difficulty = int(parts[9])
                    if difficulty == 0:
                        obj_categories_Fair.add(category)

# 将类别转换为列表并排序，以便分配索引
obj_categories_DOTA = sorted(list(obj_categories_DOTA))
obj_categories_Fair = sorted(list(obj_categories_Fair))  
# 建立类别到索引的映射
category_to_id_DOTA = {category: idx for idx, category in enumerate(obj_categories_DOTA)}

# NOTE: 对于FAIR1M, 合并飞机和船只的类别
category_to_id_Fair = {category: idx for idx, category in enumerate(obj_categories_Fair)}

root_path_DOTA = '/data/DOTA-v2.0/split_ss_dota/'
root_path_Fair = '/data/FAIR1M1.0/split_ss_fair1m/'


output_jsonl_file1 = "xxx/OD_dota2.0_sample_valid_data_train.jsonl"
output_jsonl_file2 = "xxx/OD_fair1m_sample_valid_data_train.jsonl"

dataset_name1='DOTA2.0'
question_answers1 = Process_Dataset(root_path_DOTA, 
                                   obj_categories=obj_categories_DOTA,
                                   category_to_id=category_to_id_DOTA, 
                                   type='train')

# image_name, question, answers

dataset_name2='FAIR1M'
question_answers2 = Process_Dataset(root_path_Fair, 
                                   obj_categories=obj_categories_Fair,
                                   category_to_id=category_to_id_Fair, 
                                   type='train')

# NOTE: 设置采样数量
DOTA_sample_num = 20000
FAIR_sample_num = 40000

question_answers1 = random.sample(question_answers1, DOTA_sample_num)
question_answers2 = random.sample(question_answers2, FAIR_sample_num)

count1 = 1
category_counter1 = Counter()
with open(output_jsonl_file1, 'w') as f:
    for question_answer in question_answers1:
        img_name, q, a = question_answer
        question_dict = {
            "id": count1,
            "source": dataset_name1,
            "image": f"{img_name}",
            "question": q,
            "answer": a
        }
        # 增加问题计数器
        count1 += 1
        # 写入文件
        f.write(json.dumps(question_dict) + '\n')

print(f'Total DOTA train count: {count1}')  


count2 = 1
category_counter2 = Counter()
with open(output_jsonl_file2, 'a') as f:
    for question_answer in question_answers2:
        img_name, q, a = question_answer
        # 创建对话字典
        question_dict = {
            "id": count2,  # 使用全局计数器作为问题ID
            "source": dataset_name2,
            "image": f"{img_name}",
            "question": q,
            "answer": a
        }
        # 增加问题计数器
        count2 += 1
        # 写入文件
        f.write(json.dumps(question_dict) + '\n')

print(f'Total FAIR1M train count: {count2}') 