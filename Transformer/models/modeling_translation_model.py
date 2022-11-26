import torch.nn as nn
import torch
import math
from Translation.Transformer.models.transformer import Transformer


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
    def __init__(self, args, src_vocab, tgt_vocab, dropout=0.1):
        super(TranslationModel, self).__init__()
        self.args = args
        self.src_embedding = nn.Embedding(len(src_vocab), self.args.d_model, padding_idx=2)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), self.args.d_model, padding_idx=2)
        self.positional_encoding = PositionalEncoding(args.d_model, dropout, device=args.device,
                                                      max_len=args.max_length)

        # self.transformer = nn.Transformer(args.d_model, dropout=dropout, batch_first=True)
        self.transformer = Transformer(args.d_model, dropout=dropout, batch_first=True)

        self.predictor = nn.Linear(args.d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        tgt_mask = Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(self.args.device)
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)

        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        return tokens == 2
