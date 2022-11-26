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


def seg_data():
    population = 10000000
    data = range(population)
    random.seed(1234)
    random_index = random.sample(data, 1000000)
    print(random_index)
    f_en = open(r'D:\dataset\MT\AI Challenger Translation 2017\dataset\train.en', 'r', encoding='utf-8')
    f_zh = open(r'D:\dataset\MT\AI Challenger Translation 2017\dataset\train.zh', 'r', encoding='utf-8')

    en_list = []
    zh_list = []
    en_lines = f_en.readlines()
    zh_lines = f_zh.readlines()

    for en_line in en_lines:
        en_line = en_line.strip()
        en_list.append(en_line)

    for zh_line in zh_lines:
        zh_line = zh_line.strip()
        zh_list.append(zh_line)

    w_en = open(r'D:\dataset\MT\AI Challenger Translation 2017\dataset\train_1M.en', 'a', encoding='utf-8')
    w_zh = open(r'D:\dataset\MT\AI Challenger Translation 2017\dataset\train_1M.zh', 'a', encoding='utf-8')

    for idx, en in enumerate(en_list):
        if idx in random_index:
            w_en.write(en + '\n')

    for idx, zh in enumerate(zh_list):
        if idx in random_index:
            w_zh.write(zh + '\n')


if __name__ == '__main__':
    # print(en_tokenizer("I'm a English tokenizer."))
    # print(zh_tokenizer("你好世界。"))
    seg_data()
