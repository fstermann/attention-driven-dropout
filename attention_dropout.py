from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class AttentionDropout(nn.Module):
    """Attention Dropout for input ids.

    Note: This is only used in the unsupervised settings, where sentences are used in pairs.
    In the supervised case, using attention dropout is not necessary, as the the preprocessed
    examples are used.
    """
    def __init__(
        self,
        model=None,
        n_dropout: int = 3,
        min_text_length: int = 10,
        dynamic_length: bool = False,
    ):
        super().__init__()
        self.model = model
        self.n_dropout = n_dropout
        self.min_text_length = min_text_length
        self.dynamic_length = dynamic_length
        # Load the pre-trained attention model
        # self.attention_model = BertModel.from_pretrained(model_type)

    def forward(self, input_ids):
        if not self.training:
            return input_ids
        # Use the pre-trained attention model to calculate the attention probabilities
        output: BaseModelOutputWithPoolingAndCrossAttentions = self.model(
            input_ids, attention_mask=input_ids > 0, output_attentions=True,
        )
        # output.attentions.shape: (n_layers, batch_size, num_heads, sequence_length, sequence_length)

        batch_size = len(output.attentions[0])
        seq_length = len(output.attentions[0][0][0])
        stacked = torch.stack(output.attentions).swapaxes(0, 1)

        n_dropout = seq_length // 10 if self.dynamic_length else self.n_dropout

        # Replace masked attention with 1, to avoid 0 in min
        stacked[stacked == 0] = 1
        sums = torch.sum(stacked, dim=-2).sum(1).sum(1)
        # TODO Change to sum(dim=(...)) --> Maybe swapaxes is not needed
        min_indices = sums.topk(n_dropout, dim=1, largest=False).indices

        # Drop min index in every second sentence
        input_ids = input_ids.clone()
        for i in range(1, batch_size, 2):
            sequence = input_ids[i].clone()
            n_tokens = len(sequence[sequence > 0])
            ind = min_indices[i]

            if self.dynamic_length:
                n_dropout = n_tokens // 10
                ind = ind[:n_dropout]

                if len(ind) == 0:
                    continue
            else:
                # Only replace words if there are more than min_text_length words in the sequence
                if n_tokens < self.min_text_length:
                    continue

            sequence[ind] = 0
            sequence_tokens = sequence[sequence > 0] # Without padding tokens

            # Shift tokens to the left
            ind = torch.arange(0, len(sequence_tokens), device=input_ids.device)
            result = torch.zeros_like(sequence, device=input_ids.device)
            input_ids[i] = result.scatter(0, ind, sequence_tokens)

        return input_ids.detach()

        