from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.roberta.modeling_roberta import RobertaModel


class AttentionDropout(nn.Module):
    """Attention Dropout for input ids.

    This layer is used to drop tokens from the input ids, based on their summed attention.

    Peusdo-code:

    1. Pass input_ids through the specified model to get attention scores
    2. Sum up the attention scores for each token, each batch (over the layers and heads)
            - Attention scores are of shape (num_layers, batch_size, num_heads, seq_len, seq_len)
            - We are summing up over the first, third and fourth dimension
    3. If dynamic_dropout is True:
            - Pick top k tokenswith the lowest summed attention scores, 
            with k depending on the number of input tokens (<10 -> k=1, <20 -> k=2, <30 -> k=3, ...)
    3. If dynamic_dropout is False:
            - Pick top 'n_dropout' (1, 2, 3, ...) tokens with the lowest summed attention scores
    4. Remove the picked tokens from the input_ids
            - Only if there are at least (>=) 'min_tokens' tokens in the input_ids
            - Actually the input is altered in a way such that tokens are shifted to the left and padded with 0s
    5. Return the altered input_ids

    Note: This should only be used in the unsupervised settings, where sentences are used in pairs.
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


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return input_ids
        
        # Use the pre-trained attention model to calculate the attention probabilities
        output: BaseModelOutputWithPoolingAndCrossAttentions = self.model(
            input_ids, attention_mask=input_ids > 0, output_attentions=True,
        )
        # output.attentions.shape: (n_layers, batch_size, num_heads, sequence_length, sequence_length)

        attentions = torch.stack(output.attentions)
        _, batch_size, _, _, seq_length = attentions.shape

        n_dropout = seq_length // 10 if self.dynamic_dropout else self.n_dropout

        # Replace masked attention with 1, to avoid 0 in min
        attentions[attentions == 0] = 1
        attention_sums = torch.sum(attentions, dim=(0, 2, 3)) # Sum over layers, heads and sequence length
        min_indices = attention_sums.topk(n_dropout, dim=1, largest=False).indices

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

class RandomDropout(nn.Module):

    def __init__(self, n_dropout: int = 1, min_tokens: int = 10, dynamic_dropout: bool = False) -> None:
        super().__init__()
        self.n_dropout = n_dropout
        self.min_tokens = min_tokens
        self.dynamic_dropout = dynamic_dropout

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return input_ids

        batch_size = input_ids.shape[0]

        input_ids = input_ids.clone()
        for i in range(1, batch_size, 2):
            sequence = input_ids[i].clone()
            n_tokens = len(sequence[sequence > 0])
            n_dropout = n_tokens // 10 if self.dynamic_dropout else self.n_dropout

            # Generate random sample of indices of length n_dropout
            ind = torch.randperm(n_tokens, device=input_ids.device)[:n_dropout]

            if self.dynamic_dropout:
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
        