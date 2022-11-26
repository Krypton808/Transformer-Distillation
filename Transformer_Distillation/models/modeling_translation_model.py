import torch.nn as nn
import torch
import math
from Translation.Transformer_Distillation.training.utils import from_json_file
from Translation.Transformer_Distillation.models.transformer_for_distillation import Transformer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TranslationModel(nn.Module):
    def __init__(self, args, src_vocab, tgt_vocab, dropout=0.1, is_student=False):
        super(TranslationModel, self).__init__()
        self.args = args
        self.is_student = is_student

        if self.is_student:
            config = from_json_file(args.student_transformer_config)
            d_model = config['d_model']
            self.fit_dense_encoder = nn.Linear(d_model, args.d_model)
            self.fit_dense_decoder = nn.Linear(d_model, args.d_model)

            self.transformer = Transformer(d_model=config['d_model'], num_encoder_layers=config['num_encoder_layers'],
                                           num_decoder_layers=config['num_decoder_layers'],
                                           dim_feedforward=config['dim_feedforward'], dropout=dropout, batch_first=True)
        else:
            d_model = self.args.d_model
            self.transformer = Transformer(d_model, dropout=dropout, batch_first=True)

        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
        self.positional_encoding = PositionalEncoding(d_model, dropout, device=args.device,
                                                      max_len=args.max_length)

        # self.transformer = nn.Transformer(args.d_model, dropout=dropout, batch_first=True)

        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        tgt_mask = Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(self.args.device)
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)

        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        out, all_encoder_layers, all_encoder_atts, all_decoder_layers, all_decoder_atts, all_decoder_atts_2 = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        if self.is_student:
            tmp = []
            for encoder_layer in all_encoder_layers:
                tmp.append(self.fit_dense_encoder(encoder_layer))
            all_encoder_layers = tmp

            tmp = []
            for decoder_layer in all_decoder_layers:
                tmp.append(self.fit_dense_decoder(decoder_layer))
            all_decoder_layers = tmp


        # if self.is_student:
        #     print('student')
        #
        # print(len(all_encoder_atts))
        # print(len(all_encoder_atts[0]))
        # print(len(all_encoder_atts[0][0]))
        # print('*' * 50)

        return out, all_encoder_layers, all_encoder_atts, all_decoder_layers, all_decoder_atts, all_decoder_atts_2

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        return tokens == 2
