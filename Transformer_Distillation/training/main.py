import sys

sys.path.insert(0, "./")
sys.path.append("./")

import argparse
import os

import torch
from utils import get_row_count, eval_dataset_list, set_seed, init_logger
from data_loader import TranslationDataset
from trainer import Trainer_for_distillation
from trainer_2 import Trainer_for_distillation

print(torch.cuda.is_available())


def main(args):
    init_logger()
    set_seed(args)

    en_row_count = get_row_count(args.en_filepath)
    zh_row_count = get_row_count(args.zh_filepath)
    assert en_row_count == zh_row_count, "英文和中文文件行数不一致！"

    en_eval_row_count = get_row_count(args.en_eval_filepath)
    zh_eval_row_count = get_row_count(args.zh_eval_filepath)
    assert en_eval_row_count == zh_eval_row_count, "英文和中文文件行数不一致！"

    print("句子数量为：", en_row_count)
    print("句子最大长度为：", args.max_length)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print("batch_size:", args.batch_size)
    print("每{}步保存一次模型".format(args.save_after_step))
    print("Device:", device)

    en_vocab_file = args.work_dir + "/vocab_en.pt"

    if args.use_cache and os.path.exists(en_vocab_file):
        en_vocab = torch.load(en_vocab_file, map_location="cpu")
    # else:
    #     en_vocab = build_vocab_from_iterator(
    #         # 传入一个可迭代的token列表。例如[['i', 'am', ...], ['machine', 'learning', ...], ...]
    #         yield_en_tokens(en_filepath=args.en_filepath, row_count=en_row_count),
    #         # 最小频率为2，即一个单词最少出现两次才会被收录到词典
    #         min_freq=2,
    #         # 在词典的最开始加上这些特殊token
    #         specials=["<s>", "</s>", "<pad>", "<unk>"],
    #     )
    #     # 设置词典的默认index，后面文本转index时，如果找不到，就会用该index填充
    #     en_vocab.set_default_index(en_vocab["<unk>"])
    #     # 保存缓存文件
    #     if args.use_cache:
    #         torch.save(en_vocab, en_vocab_file)

    print("英文词典大小:", len(en_vocab))
    print(dict((i, en_vocab.lookup_token(i)) for i in range(10)))

    zh_vocab_file = args.work_dir + "/vocab_zh.pt"
    if args.use_cache and os.path.exists(zh_vocab_file):
        zh_vocab = torch.load(zh_vocab_file, map_location="cpu")
    # else:
    #     zh_vocab = build_vocab_from_iterator(
    #         yield_zh_tokens(zh_filepath=zh_vocab_file, row_count=zh_row_count),
    #         min_freq=1,
    #         specials=["<s>", "</s>", "<pad>", "<unk>"],
    #     )
    #     zh_vocab.set_default_index(zh_vocab["<unk>"])
    #     torch.save(zh_vocab, zh_vocab_file)

    print("中文词典大小:", len(zh_vocab))
    print(dict((i, zh_vocab.lookup_token(i)) for i in range(10)))
    train_dataset = TranslationDataset(args, args.en_filepath, args.zh_filepath, en_vocab, zh_vocab, en_row_count)
    print(train_dataset.__getitem__(0))
    eval_dataset = TranslationDataset(args, args.en_eval_filepath, args.zh_eval_filepath, en_vocab, zh_vocab,
                                      en_eval_row_count, lang_suffix='_eval')
    print(eval_dataset.__getitem__(0))
    eval_dataset = eval_dataset_list(args.en_eval_filepath, args.zh_eval_filepath)

    # trainer = Trainer_for_distillation(args, en_vocab, zh_vocab, train_dataset, eval_dataset)
    # trainer.train()
    # trainer.evaluate()
    trainer_2 = Trainer_for_distillation(args, en_vocab, zh_vocab, train_dataset, eval_dataset)
    # trainer_2.train()
    trainer_2.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir',
                        default=r'D:\LearningSet\UM\NLP\project\tinybert_experiment\Translation\experiment', type=str)
    parser.add_argument('--output_model_dir',
                        default=r'D:\LearningSet\UM\NLP\project\translation_checkpoint\student_model_checkpoint',
                        type=str)
    # parser.add_argument('--model_checkpoint', default=r'D:\LearningSet\UM\NLP\project\checkpoint\model_190.pt', type=str)
    parser.add_argument('--teacher_model_checkpoint',
                        default=r'D:\LearningSet\UM\NLP\project\translation_checkpoint\teacher_model_checkpoint\model_390625.pt',
                        type=str)
    # parser.add_argument('--student_model_checkpoint',
    #                     default=r'D:\LearningSet\UM\NLP\project\translation_checkpoint\tmp_checkpoint\model_156250.pt',
    #                     type=str)

    parser.add_argument('--student_model_checkpoint',
                        default=r'D:\LearningSet\UM\NLP\project\translation_checkpoint\student_model_checkpoint\model_328125.pt',
                        type=str)

    parser.add_argument('--final_model_checkpoint',
                        default=r'D:\LearningSet\UM\NLP\project\translation_checkpoint\student_model_checkpoint\model_328125.pt',
                        type=str)

    parser.add_argument('--en_filepath', default=r'D:\dataset\MT\AI Challenger Translation 2017\dataset\train_1M.en',
                        type=str)
    parser.add_argument('--zh_filepath', default=r'D:\dataset\MT\AI Challenger Translation 2017\dataset\train_1M.zh',
                        type=str)

    parser.add_argument('--en_eval_filepath', default=r'D:\dataset\MT\AI Challenger Translation 2017\dataset\train_1M.en',
                        type=str)
    parser.add_argument('--zh_eval_filepath', default=r'D:\dataset\MT\AI Challenger Translation 2017\dataset\train_1M.zh',
                        type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int)

    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--max_length', default=72, type=int)
    parser.add_argument('--save_after_step', default=5000, type=int)
    parser.add_argument('--use_cache', default=True, type=bool)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--warmup_steps', default=1000, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--logging_steps', default=15625, type=int)
    parser.add_argument("--patience", default=6, type=int)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--do_eval", default=False)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--student_transformer_config',
                        default=r'D:\LearningSet\UM\NLP\project\tinybert_experiment\Translation\Transformer_Distillation\models\student_transformer_config.json')

    args = parser.parse_args()
    main(args)
