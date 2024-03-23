import json
import os
import re
from typing import List

import numpy as np
import random

import torch
from tqdm import tqdm


def get_id_and_prob(spans, offset_map):
    prompt_length = 0
    for i in range(1, len(offset_map)):
        if offset_map[i] != [0, 0]:
            prompt_length += 1
        else:
            break

    for i in range(1, prompt_length + 1):
        offset_map[i][0] -= (prompt_length + 1)
        offset_map[i][1] -= (prompt_length + 1)

    sentence_id = []
    prob = []
    for start, end in spans:
        prob.append(start[1] * end[1])
        sentence_id.append(
            (offset_map[start[0]][0], offset_map[end[0]][1]))
    return sentence_id, prob
def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely, reference to
    "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r'([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")
def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if not (0x0021 <= code and code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def add_negative_samples(entities, texts, prompts, label_set, negative_ratio):
    with tqdm(total=len(prompts)) as bar:
        for i, prompt in enumerate(prompts):
            negative_sample = []
            # ^ is the symmetric difference which means the item is
            # either in a or in b, but not in both
            # redundant_list has labels that are not included in the current entity
            redundant_list = list(set(label_set) ^ set(prompt))
            redundant_list.sort()
            if len(entities[i]) == 0:
                continue
            # the meaning of negative_ratio is, if negative_ratio is 2,
            # and the current entity has 3 labels, then we need to create 2*3=6 negative samples.
            # the negative sample should be chosen from redundant_list,
            # it's possible that there are only 2 labels in the redundant list,
            # but what we actually need is 10. In this case, we can only create 2 negative samples
            required_label_num = len(entities[i]) * negative_ratio
            actual_len = min(required_label_num, len(redundant_list))
            indices = random.sample(range(0, len(redundant_list)), actual_len)
            for index in indices:
                negative_result={"content": texts[i], "result_list": [], "prompt": redundant_list[index]}
                negative_sample.append(negative_result)
            entities[i].extend(negative_sample)
            bar.update(1)
    return entities


def convert(data: List[str], negative_ratio: int):
    texts = []
    entity_all = []
    relations = []
    entity_prompts = [] # the len equals len(data)
    relation_prompts = []
    entity_label_set = []
    entity_name_set = []
    predicate_set = []
    print("Start converting......")
    with tqdm(total=len(data)) as bar:
        for line in data:
            item = json.loads(line)
            txt, entities, relations = item["text"], item["entities"], item["relations"]
            texts.append(txt)
            entity_list = []  # entities in a single line
            entity_prompt = []
            entity_map_label2names = {}
            entity_map_id2name = {}  # id to entity name, it will be used if we need to extract relations
            for entity in entities:
                entity_name = txt[entity["start_offset"]:entity["end_offset"]]
                entity_map_id2name[entity["id"]] = {
                    "name": entity_name,
                    "start": entity["start_offset"],
                    "end": entity["end_offset"]
                }
                entity_label = entity["label"]
                result = {"text": entity_name, "start": entity["start_offset"], "end": entity["end_offset"]}
                if entity_label not in entity_map_label2names:
                    entity_map_label2names[entity_label] = {
                        "content": txt,
                        "result_list": [result],
                        "prompt": entity_label
                    }
                else:
                    entity_map_label2names[entity_label]["result_list"].append(result)
                if entity_label not in entity_label_set:
                    entity_label_set.append(entity_label)
                if entity_name not in entity_name_set:
                    entity_name_set.append(entity_name)
                entity_prompt.append(entity_label)
            for v in entity_map_label2names.values():
                entity_list.append(v)

            entity_all.append(entity_list)
            entity_prompts.append(entity_prompt)
            # TODO, extrac relations in the future when it's necessary
            bar.update(1)

    # add negative samples
    print("Start adding negative samples......")
    results = add_negative_samples(entity_all, texts, entity_prompts, entity_label_set, negative_ratio)
    return results
