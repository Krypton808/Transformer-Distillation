import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_row_count, en_tokenizer, yield_en_tokens, zh_tokenizer, yield_zh_tokens
from torch.nn.functional import pad
from tqdm import tqdm


class TranslationDataset(Dataset):
    def __init__(self, args, en_filepath, zh_filepath, en_vocab, zh_vocab, row_count, lang_suffix=''):
        self.args = args
        self.row_count = row_count
        self.en_tokens = self.load_tokens(en_filepath, en_tokenizer, en_vocab, "构建英文tokens", 'en' + lang_suffix)
        self.zh_tokens = self.load_tokens(zh_filepath, zh_tokenizer, zh_vocab, "构建中文tokens", 'zh' + lang_suffix)

    def __getitem__(self, index):
        return self.en_tokens[index], self.zh_tokens[index]

    def __len__(self):
        return self.row_count

    def load_tokens(self, file, tokenizer, vocab, desc, lang):
        """
        加载tokens，即将文本句子们转换成index们。
        :param file: 文件路径，例如"./dataset/train.en"
        :param tokenizer: 分词器，例如en_tokenizer函数
        :param vocab: 词典, Vocab类对象。例如 en_vocab
        :param desc: 用于进度显示的描述，例如：构建英文tokens
        :param lang: 语言。用于构造缓存文件时进行区分。例如：’en‘
        :return: 返回构造好的tokens。例如：[[6, 8, 93, 12, ..], [62, 891, ...], ...]
        """

        # 定义缓存文件存储路径
        cache_file = self.args.work_dir + "/tokens_list.{}.pt".format(lang)
        # 如果使用缓存，且缓存文件存在，则直接加载
        if self.args.use_cache and os.path.exists(cache_file):
            print(f"正在加载缓存文件{cache_file}, 请稍后...")
            return torch.load(cache_file, map_location="cpu")
            # 从0开始构建，定义tokens_list用于存储结果

        tokens_list = []
        # 打开文件
        with open(file, encoding='utf-8') as file:
            # 逐行读取
            for line in tqdm(file, desc=desc, total=self.row_count):
                # 进行分词
                tokens = tokenizer(line)
                # 将文本分词结果通过词典转成index
                tokens = vocab(tokens)
                # append到结果中
                tokens_list.append(tokens)
            # 保存缓存文件
        if self.args.use_cache:
            print('保存')
            torch.save(tokens_list, cache_file)
        return tokens_list


def collate_fn(batch):
    """
    将dataset的数据进一步处理，并组成一个batch。
    :param batch: 一个batch的数据，例如：
                  [([6, 8, 93, 12, ..], [62, 891, ...]),
                  ....
                  ...]
    :return: 填充后的且等长的数据，包括src, tgt, tgt_y, n_tokens
             其中src为原句子，即要被翻译的句子
             tgt为目标句子：翻译后的句子，但不包含最后一个token
             tgt_y为label：翻译后的句子，但不包含第一个token，即<bos>
             n_tokens：tgt_y中的token数，<pad>不计算在内。
    """

    # 定义'<bos>'的index，在词典中为0，所以这里也是0
    bs_id = torch.tensor([0])
    # 定义'<eos>'的index
    eos_id = torch.tensor([1])
    # 定义<pad>的index
    pad_id = 2

    # 用于存储处理后的src和tgt
    src_list, tgt_list = [], []

    # 循环遍历句子对儿
    for (_src, _tgt) in batch:
        """
        _src: 英语句子，例如：`I love you`对应的index
        _tgt: 中文句子，例如：`我 爱 你`对应的index
        """

        processed_src = torch.cat(
            # 将<bos>，句子index和<eos>拼到一块
            [
                bs_id,
                torch.tensor(
                    _src,
                    dtype=torch.int64,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    _tgt,
                    dtype=torch.int64,
                ),
                eos_id,
            ],
            0,
        )

        """
        将长度不足的句子进行填充到max_padding的长度的，然后增添到list中

        pad：假设processed_src为[0, 1136, 2468, 1349, 1]
             第二个参数为: (0, 72-5)
             第三个参数为：2
        则pad的意思表示，给processed_src左边填充0个2，右边填充67个2。
        最终结果为：[0, 1136, 2468, 1349, 1, 2, 2, 2, ..., 2]
        """
        src_list.append(
            pad(
                processed_src,
                (0, 72 - len(processed_src),),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, 72 - len(processed_tgt),),
                value=pad_id,
            )
        )

    # 将多个src句子堆叠到一起
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    # tgt_y是目标句子去掉第一个token，即去掉<bos>
    tgt_y = tgt[:, 1:]
    # tgt是目标句子去掉最后一个token
    tgt = tgt[:, :-1]

    # 计算本次batch要预测的token数
    n_tokens = (tgt_y != 2).sum()

    # 返回batch后的结果
    return src, tgt, tgt_y, n_tokens
