import json
import random
import numpy as np
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer


def get_row_count(filepath):
    count = 0
    for _ in open(filepath, encoding='utf-8'):
        count += 1
    return count


def en_tokenizer(line):
    """
    定义英文分词器，后续也要使用
    :param line: 一句英文句子，例如"I'm learning Deep learning."
    :return: subword分词后的记过，例如：['i', "'", 'm', 'learning', 'deep', 'learning', '.']
    """
    # 使用bert进行分词，并获取tokens。add_special_tokens是指不要在结果中增加‘<bos>’和`<eos>`等特殊字符

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    tokenized = tokenizer.encode(line, add_special_tokens=False)

    to_tokens = tokenizer.convert_ids_to_tokens(tokenized)
    return to_tokens


def yield_en_tokens(en_filepath, row_count):
    """
    每次yield一个分词后的英文句子，之所以yield方式是为了节省内存。
    如果先分好词再构造词典，那么将会有大量文本驻留内存，造成内存溢出。
    """
    file = open(en_filepath, encoding='utf-8')
    print("-------开始构建英文词典-----------")
    for line in tqdm(file, desc="构建英文词典", total=row_count):
        yield en_tokenizer(line)
    file.close()


def zh_tokenizer(line):
    """
    定义中文分词器
    :param line: 中文句子，例如：机器学习
    :return: 分词结果，例如['机','器','学','习']
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    tokenized = tokenizer.encode(line, add_special_tokens=False)

    to_tokens = tokenizer.convert_ids_to_tokens(tokenized)

    # list(line.strip().replace(" ", ""))

    return to_tokens


def yield_zh_tokens(zh_filepath, row_count):
    file = open(zh_filepath, encoding='utf-8')
    for line in tqdm(file, desc="构建中文词典", total=row_count):
        yield zh_tokenizer(line)
    file.close()


def eval_dataset_list(en_filepath, zh_filepath):
    f_en = open(en_filepath, 'r', encoding='utf-8')
    f_zh = open(zh_filepath, 'r', encoding='utf-8')
    lines = f_en.readlines()

    list_en = []
    list_zh = []

    for line in lines:
        list_en.append(line.strip())

    lines = f_zh.readlines()

    for line in lines:
        list_zh.append(line.strip())

    return list_en, list_zh


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def from_dict(json_object):
    config = {}
    for key, value in json_object.items():
        config[key] = value
    return config


def from_json_file(json_file):
    with open(json_file, "r", encoding='utf-8') as reader:
        text = reader.read()
    return from_dict(json.loads(text))


if __name__ == '__main__':
    # print(en_tokenizer("I'm a English tokenizer."))
    # print(zh_tokenizer("你好世界。"))

    config = from_json_file(
        r'D:\LearningSet\UM\NLP\project\tinybert_experiment\Translation\Transformer_Distillation\models\student_transformer_config.json')
    print(config)
