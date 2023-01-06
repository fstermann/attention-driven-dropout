from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.roberta.modeling_roberta import RobertaModel


class AttentionDropout(nn.Module):
    """Attention Dropout for input ids.

    Note: This is only used in the unsupervised settings, where sentences are used in pairs.
    In the supervised case, using attention dropout is not necessary, as the the preprocessed
    examples are used.

    Args:
        model (str | BertModel | RobertaModel): The pre-trained model to use. Either a string
            with the name of the model or a model object.
        n_dropout (int, optional): The number of tokens to drop. Defaults to 1.
        min_tokens (int, optional): The minimum number of tokens present in a sequence to be 
            able to drop tokens. Defaults to 10.
        dynamic_dropout (bool, optional): If True, the number of tokens to drop is calculated
            dynamically from the sequence length (seq_length // 10). Defaults to False.

    """
    def __init__(
        self,
        model: str | BertModel | RobertaModel,
        *,
        n_dropout: int = 1,
        min_tokens: int = 10,
        dynamic_dropout: bool = False,
    ):
        super().__init__()
        if isinstance(model, str):
            if model == "bert-base-uncased":
                self.model = BertModel.from_pretrained(model)
            elif model == "roberta-base":
                self.model = RobertaModel.from_pretrained(model)
            else:
                raise ValueError(f"Model {model} is not supported.")
        else:
            self.model = model

        self.n_dropout = n_dropout
        self.min_tokens = min_tokens
        self.dynamic_dropout = dynamic_dropout


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
        stacked = torch.stack(output.attentions)

        n_dropout = seq_length // 10 if self.dynamic_dropout else self.n_dropout

        # Replace masked attention with 1, to avoid 0 in min
        stacked[stacked == 0] = 1
        sums = torch.sum(stacked, dim=(0, 2, 3)) # Sum over layers, heads and sequence length
        min_indices = sums.topk(n_dropout, dim=1, largest=False).indices

        # Drop min index in every second sentence
        input_ids = input_ids.clone()
        for i in range(1, batch_size, 2):
            sequence = input_ids[i].clone()
            n_tokens = len(sequence[sequence > 0])
            ind = min_indices[i]

            if self.dynamic_dropout:
                n_dropout = n_tokens // 10
                ind = ind[:n_dropout]

                if len(ind) == 0:
                    continue
            else:
                # Only replace words if there are more than min_tokens words in the sequence
                if n_tokens < self.min_tokens:
                    continue

            sequence[ind] = 0
            sequence_tokens = sequence[sequence > 0] # Without padding tokens

            # Shift tokens to the left
            ind = torch.arange(0, len(sequence_tokens), device=input_ids.device)
            result = torch.zeros_like(sequence, device=input_ids.device)
            input_ids[i] = result.scatter(0, ind, sequence_tokens)

        return input_ids.detach()

        