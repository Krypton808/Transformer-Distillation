import torch
import logging
import time
from Translation.Transformer_Distillation.models.modeling_translation_model import TranslationModel
from data_loader import collate_fn
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_cosine_schedule_with_warmup
from torch.nn import CrossEntropyLoss, MSELoss
from loss import TranslationLoss
from tqdm import tqdm, trange
import sacrebleu
from utils import en_tokenizer

logger = logging.getLogger(__name__)
w = open(
    r'D:\LearningSet\UM\NLP\project\tinybert_experiment\Translation\experiment\log\transformer_distill_2_log.txt',
    'a',
    encoding='utf-8')


class Trainer_for_distillation(object):
    def __init__(self, args, en_vocab, zh_vocab, train_dataset, eval_dataset):
        self.args = args
        self.device = args.device
        self.train_dataset = train_dataset
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab
        self.eval_dataset = eval_dataset

        self.criteria = TranslationLoss()

        self.best_score = -1e+10
        self.patience = args.patience
        self.early_stopping_counter = 0
        self.do_early_stop = False

        if not args.do_eval:
            self.teacher_model = TranslationModel(args, en_vocab, zh_vocab)
            self.load_model(self.args.teacher_model_checkpoint)
            self.teacher_model.to(self.device)

        self.student_model = TranslationModel(args, en_vocab, zh_vocab, is_student=True)

        if args.student_model_checkpoint:
            self.load_model(self.args.student_model_checkpoint, is_student=True)

        self.student_model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size,
                                      collate_fn=collate_fn)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        src, tgt, tgt_y, n_tokens = next(iter(train_dataloader))
        src, tgt, tgt_y = src.to(self.device), tgt.to(self.device), tgt_y.to(self.device)
        print("src.size:", src.size())
        print("tgt.size:", tgt.size())
        print("tgt_y.size:", tgt_y.size())
        print("n_tokens:", n_tokens)

        for n, p in self.student_model.named_parameters():
            print(n)

        param_optimizer = list(self.student_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args.learning_rate)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        global_step = 0
        tr_loss = 0
        self.student_model.zero_grad()

        if self.args.final_model_checkpoint:
            global_step = int(
                self.args.final_model_checkpoint.replace(
                    r"D:\LearningSet\UM\NLP\project\translation_checkpoint\student_model_checkpoint\model_",
                    "").replace(
                    ".pt",
                    ""))
            print('global_step: ' + str(global_step))
            # logger.info('global_step: ' + global_step)

        train_iterator = trange(int(self.args.num_train_epochs), desc='Epoch')

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')

            for step, batch in enumerate(epoch_iterator):
                self.student_model.train()
                src, tgt, tgt_y, n_tokens = batch
                src, tgt, tgt_y = src.to(self.device), tgt.to(self.device), tgt_y.to(self.device)

                out, student_encoder_reps, student_encoder_atts, student_decoder_reps, student_decoder_atts, student_decoder_atts_2 = self.student_model(
                    src, tgt)
                out = self.student_model.predictor(out)

                loss = self.criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1),
                                     device=self.device) / n_tokens

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.student_model.zero_grad()
                    global_step += 1

                    if global_step % 250 == 0:
                        print('loss: ' + str(loss))
                        # logger.info("loss: " + loss)

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        # logger.info("*" * 50)
                        # logger.info('Saving model')
                        # logger.info("loss: " + loss)
                        # logger.info("*" * 50)
                        print("*" * 50)
                        print('Saving model')
                        print('global_step: ' + str(global_step))
                        w.write('global_step: ' + str(global_step) + '\n')
                        print("loss: " + str(loss))
                        w.write('loss: ' + str(loss) + '\n')
                        print('\n')
                        print("*" * 50)
                        # print("loss: " + loss)
                        self.save_model(global_step)
                        # score = self.evaluate()
                        #
                        # logger.info("*" * 50)
                        # logger.info("current step score for metric_key_for_early_stop: {}".format(score))
                        # logger.info("best score for metric_key_for_early_stop: {}".format(self.best_score))
                        # logger.info("*" * 50)
                        #
                        # if score > self.best_score:
                        #     self.best_score = score
                        #     self.early_stopping_counter = 0
                        #     self.save_model(global_step)
                        #
                        # else:
                        #     self.early_stopping_counter += 1
                        #     if self.early_stopping_counter >= self.patience:
                        #         self.do_early_stop = True
                        #
                        #         logger.info("best score is {}".format(self.best_score))
                        #
                        # if self.do_early_stop:
                        #     break

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

                if self.do_early_stop:
                    epoch_iterator.close()
                    break

                time.sleep(0.5)

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

            if self.do_early_stop:
                epoch_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self):
        # logger.info("***** Running evaluation on eval dataset *****")
        # logger.info("  Num examples = %d", len(self.eval_dataset))
        # logger.info("  Batch size = %d", self.args.eval_batch_size)
        print("***** Running evaluation on eval dataset *****")
        print("  Num examples = %d", len(self.eval_dataset))
        print("  Batch size = %d", self.args.eval_batch_size)
        w = open('distill_version_eval_result.txt', 'a', encoding='utf-8')

        self.student_model.eval()
        src_list = self.eval_dataset[0]
        tgt_groundtruth_list = self.eval_dataset[1]

        tgt_list = []
        for src in tqdm(src_list):
            src = torch.tensor([0] + self.en_vocab(en_tokenizer(src)) + [1]).unsqueeze(0).to(self.device)
            tgt = torch.tensor([[0]]).to(self.device)

            for i in range(self.args.max_length):
                out, student_encoder_reps, student_encoder_atts, student_decoder_reps, student_decoder_atts, student_decoder_atts_2 = self.student_model(
                    src, tgt)
                predict = self.student_model.predictor(out[:, -1])
                y = torch.argmax(predict, dim=1)
                tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
                if y == 1:
                    break
            tgt = ''.join(self.zh_vocab.lookup_tokens(tgt.squeeze().tolist())).replace("<s>", "").replace("</s>", "")
            tgt_list.append(tgt)
            print(tgt)
            w.write(tgt + '\n')

        metric = sacrebleu.metrics.BLEU()
        score = metric.corpus_score(tgt_list, [tgt_groundtruth_list]).score

        return score

    def save_model(self, global_step):
        # logger.info("*" * 50)
        # logger.info('Saving model')
        # logger.info("*" * 50)
        print("*" * 50)
        print('Saving model')
        print("*" * 50)
        # print('Saving model')
        # torch.save(self.model, self.args.output_model_dir + f"/model_{global_step}.pt")
        torch.save(self.student_model.state_dict(), self.args.output_model_dir + f"/model_{global_step}.pt")

    def load_model(self, model_checkpoint, is_student=False):
        logger.info("*" * 50)
        logger.info('Loading model')
        logger.info("*" * 50)
        # print("*" * 50)
        # print('Loading model')
        # print("*" * 50)
        # print('Loading model')
        # model = torch.load(model_checkpoint)
        # model = torch.load(model_checkpoint)
        if is_student:
            self.student_model.load_state_dict(torch.load(model_checkpoint))
        else:
            self.teacher_model.load_state_dict(torch.load(model_checkpoint))
