# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""
import argparse
import collections
from dataclasses import field, dataclass
import torch
import torch.nn.functional as F

import omegaconf
from fairseq.models import register_model
from fairseq.models.nat import CMLMNATransformerModel
from fairseq.models.transformer import TransformerConfig
from fairseq.models.nat.cmlm_transformer import _skeptical_unmasking

from fairseq_easy_extend.dataclass.utils import gen_parser_from_dataclass
from fairseq_easy_extend.dataclass.utils import convert_omegaconf_to_namesapce


@dataclass
class CMLMTransformerConfig(TransformerConfig):
    # --- special arguments ---
    sg_length_pred: bool = field(
        default=False,
        metadata={
            "help": "stop gradients through length"
        }
    )
    pred_length_offset: bool = field(
        default=False,
        metadata={
            "help": "predict length offset"
        },
    )
    length_loss_factor: float = field(
        default=0.1,
        metadata={"help": "loss factor for length"},
    )
    ngram_predictor: int = field(
        default=1, metadata={"help": "maximum iterations for iterative refinement."},
    )
    src_embedding_copy: bool = field(
        default=False,
        metadata={
            "help": "copy source embeddings"
        },
    )
    label_smoothing: float = field(default=0.1, metadata={"help": "label smoothing"})

@register_model("cmlm_transformer_base", dataclass=CMLMTransformerConfig)
class BaseCMLMNATransformerModel(CMLMNATransformerModel):

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)

        if kwargs["sampling"]:
            _scores = self.decoder(
                normalize=False,
                prev_output_tokens=output_tokens,
                encoder_out=encoder_out,
            )

            batch_dim=_scores.size(0)
            seq_len=_scores.size(1)
            vocab_len=_scores.size(2)

            _scores = F.softmax(_scores/kwargs["temperature"], dim=-1)
            if kwargs["k"] > 0: # set scores outside top k to 0
                topk_values = _scores.topk(kwargs["k"], dim=-1)
                mask = _scores < topk_values[:, -1].unsqueeze(-1)
                _scores[mask] = 0.
            _tokens = torch.multinomial(_scores.view(-1, vocab_len), 1)
            _tokens = _tokens.view(batch_dim,seq_len).unsqueeze(-1)
            _scores = _scores.gather(-1,_tokens).squeeze(-1)
            _tokens = _tokens.squeeze(-1)
        
        else:
            _scores, _tokens = self.decoder(
                normalize=True,
                prev_output_tokens=output_tokens,
                encoder_out=encoder_out,
            ).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, CMLMTransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        if isinstance(cfg, omegaconf.DictConfig):
            cfg = convert_omegaconf_to_namesapce(cfg)
        model = super().build_model(cfg, task)
        return model