from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.roberta.modeling_roberta import RobertaModel


class AttentionDrivenDropout(nn.Module):
    """Attention-Driven Dropout for tokenized input ids in self-contrastive learning.

    This layer is used to drop tokens from the input ids, based on their
    aggregated attention scores.

    Peusdo-code:

    1. Pass input_ids through the specified model to get attention scores
    2. - If aggregation_method is "naive":
                - Sum up the attention scores for each token, each batch (over the layers and heads)
            - If aggregation_method is "rollout":
                - Use the Attention Rollout method to aggregate the attention scores
    3. - If dropout_rate is "dynamic":
                - Pick top k tokens with the lowest summed attention scores,
                with k depending on the number of input tokens, e.g. for min_tokens = 10:
                    (<10 -> k=1, <20 -> k=2, <30 -> k=3, ...)
            - If dropout_rate is "static":
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
        dropout_rate (str, optional): Dropout rate mode to figure our the number of tokens to drop. Defaults to "static".
            - "static": Drop 'n_dropout' tokens
            - "dynamic": Drop tokens based on the number of tokens in the sequence
        summation_method (str, optional): The method to use to aggregate the attention scores. Defaults to "naive".
            - "naive": Sum up the attention scores for each token, each batch (over the layers and heads)
            - "rollout": Use the Attention Rollout method to aggregate the attention scores
    """

    def __init__(
        self,
        model: str | BertModel | RobertaModel,
        *,
        n_dropout: int = 1,
        min_tokens: int = 10,
        dropout_rate: str = "static",
        summation_method: str = "naive",
    ):
        super().__init__()
        self.n_dropout = n_dropout
        self.min_tokens = min_tokens
        if dropout_rate not in ["static", "dynamic"]:
            raise ValueError(f"Dropout rate {dropout_rate} is not supported.")
        self.dropout_rate = dropout_rate

        self.model = self.get_model(model)
        logging.info(f"[ADD] model: {self.model.base_model_prefix}")

        self.padding_token = self.get_padding_token(model)
        logging.info(f"[ADD] padding_token: {self.padding_token}")

        self.aggregation_method = summation_method
        self.aggregation_func = self.get_aggregation_func(summation_method)
        logging.info(f"[ADD] aggregation_method: {self.aggregation_method}")
        logging.info(f"[ADD] aggregation_func: {self.aggregation_func}")

    @staticmethod
    def get_model(model: str | BertModel | RobertaModel) -> BertModel | RobertaModel:
        if not isinstance(model, str):
            return model
        if model == "bert-base-uncased":
            return BertModel.from_pretrained(model)
        if model == "roberta-base":
            return RobertaModel.from_pretrained(model)
        raise ValueError(f"Model {model} is not supported.")

    @staticmethod
    def get_padding_token(model: BertModel | RobertaModel) -> int:
        # Using BERT tokenizer
        if model.base_model_prefix == "bert":
            return 0
        # Using RoBERTa tokenizer
        if model.base_model_prefix == "roberta":
            return 1
        raise ValueError(f"Model {model.base_model_prefix} is not supported.")

    def get_aggregation_func(self, aggregation_method: str) -> Callable:
        AGGREGATION_CHOICES = {
            "naive": self.aggregate_attentions_naive,
            "rollout": self.aggregate_attentions_rollout,
        }
        if aggregation_method not in AGGREGATION_CHOICES:
            raise ValueError(
                f"Aggregation method {aggregation_method} is not supported."
            )

        return AGGREGATION_CHOICES[aggregation_method]

    def forward(
        self,
        input_ids: torch.Tensor,
        return_scores: bool = False,
        num_sent: int = 2,
        last_layer_only: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids (torch.Tensor): The input ids. Shape: (batch_size * num_sent, seq_length)
            return_scores (bool, optional): If True, the summed attention scores are returned.
                Defaults to False.
            num_sent (int, optional): The number of sentences per input pair. Defaults to 2.
            last_layer_only (bool, optional): If True, only the last attention layer is used in
                the aggregation of the attention scores. Defaults to False.

        Returns:
            torch.Tensor: The input ids with dropped tokens. Shape: (batch_size * num_sent, seq_length)
        """
        if not self.training:
            return input_ids

        # Use the pre-trained attention model to calculate the attention probabilities
        attention_input_ids = self.get_last_of_input_ids(input_ids, num_sent=num_sent)
        mask = attention_input_ids != self.padding_token

        output = self.model(
            attention_input_ids, attention_mask=mask, output_attentions=True
        )
        # output.attentions.shape: (n_layers, batch_size, num_heads, sequence_length, sequence_length)

        attentions = torch.stack(output.attentions)
        _, batch_size, _, _, seq_length = attentions.shape

        # Determine dropout rate
        if self.dropout_rate == "static":
            n_dropout = self.n_dropout
        elif self.dropout_rate == "dynamic":
            n_dropout = seq_length // self.min_tokens
        else:
            raise ValueError(f"Dropout rate {self.dropout_rate} is not supported.")

        # Sum over layers, heads and sequence length
        attention_sums = self.aggregation_func(
            attentions, last_layer_only=last_layer_only
        )

        min_indices = attention_sums.topk(n_dropout, dim=1, largest=False).indices

        # Drop min index in every second sentence
        input_ids = input_ids.clone()
        for i in range(batch_size):
            input_index = num_sent * (i + 1) - 1
            sequence = input_ids[input_index].clone()

            n_tokens = len(sequence[sequence != self.padding_token])
            ind = min_indices[i]

            if self.dropout_rate == "static":
                # Only replace words if there are more than min_tokens words in the sequence
                if n_tokens < self.min_tokens:
                    continue
            elif self.dropout_rate == "dynamic":
                n_dropout = n_tokens // self.min_tokens
                ind = ind[:n_dropout]

                if len(ind) == 0:
                    continue

            sequence[ind] = self.padding_token
            # Select tokens without padding tokens
            sequence_tokens = sequence[sequence != self.padding_token]

            # Shift tokens to the left
            input_ids[input_index] = F.pad(
                sequence_tokens,
                pad=(0, seq_length - len(sequence_tokens)),
                value=self.padding_token,
            )

        if return_scores:
            return input_ids.detach(), attention_sums
        return input_ids.detach()

    @staticmethod
    def get_last_of_input_ids(
        input_ids: torch.Tensor, num_sent: int = 2
    ) -> torch.Tensor:
        """Get the last sentence for each input pair.

        The input_ids x_ij look like this:
            [x(1,1), x(1,2), ..., x(1,num_sent), x(2,1), x(2,2), ..., x(2,num_sent), ...]
        The output should look like this:
            [x(1,num_sent), x(2,num_sent), ...]

        Args:
            input_ids (torch.Tensor): The input ids. Shape: (batch_size * num_sent, seq_length)
            num_sent (int, optional): The number of sentences per input pair. Defaults to 2.

        Returns:
            torch.Tensor: The set of input ids for the last sentence in each input pair.
                Shape: (batch_size // num_sent, seq_length)
        """
        split_shape = (
            input_ids.shape[0] // num_sent,
            num_sent,
            *input_ids.shape[1:],
        )
        split_input_ids = input_ids.reshape(split_shape).clone()
        return split_input_ids[:, num_sent - 1]

    @staticmethod
    def aggregate_attentions_naive(
        attentions: torch.Tensor,
        dim: tuple[int, ...] = (0, 2, 3),
        last_layer_only: bool = False,
    ) -> torch.Tensor:
        """Compute the summed attention scores for each token in the sequence.

        Args:
            attentions (torch.Tensor): The attention scores of shape (n_layers, batch_size, num_heads, seq_len, seq_len)
            dim (tuple[int, ...], optional): The dimensions to sum over. Defaults to (0, 2, 3).
            last_layer_only (bool, optional): If True, only the last attention layer is used in
                the aggregation of the attention scores. Defaults to False.

        Returns:
            torch.Tensor: The summed attention scores of shape (batch_size, seq_len)
        """
        if last_layer_only:
            attentions = attentions[-1].unsqueeze(0)

        output = torch.sum(attentions, dim=dim)

        # Replace masked attentions with Inf, to exclude them from the min
        mask = attentions[0, :, 1, 1] == 0
        output[mask] = float("inf")
        return output

    @staticmethod
    def aggregate_attentions_rollout(
        attentions: torch.Tensor,
        dim: tuple[int, ...] = (0, 2),
        last_layer_only: bool = False,
    ) -> torch.Tensor:
        """Compute the summed attention scores for each token in the sequence with the Attention Rollout method.

        Args:
            attentions (torch.Tensor): The attention scores of shape (n_layers, batch_size, num_heads, seq_len, seq_len)
            dim (tuple[int, ...], optional): The dimensions to sum over. Defaults to (0, 2).
            last_layer_only (bool, optional): If True, only the last attention layer is used in
                the aggregation of the attention scores. Defaults to False.

        Returns:
            torch.Tensor: The summed attention scores of shape (batch_size, seq_len)
        """
        residual_attentions = get_residual_attentions(attentions)
        rollout_attentions = compute_rollout_attention(
            residual_attentions, add_residual=False
        )

        if last_layer_only:
            rollout_attentions = rollout_attentions[-1].unsqueeze(0)

        output = rollout_attentions.sum(dim=dim)

        # Replace masked attentions with Inf, to exclude them from the min
        mask = attentions[0, :, 1, 1] == 0
        output[mask] = float("inf")
        return output


# ==========================================
# ==== ATTENTION FLOW/ROLLOUT FUNCTIONS ====
# ==========================================


def get_residual_attentions(
    attentions: torch.Tensor, head_dim: int = 2
) -> torch.Tensor:
    """Compute the residual attention matrices from the attention matrices of each layer.

    Args:
        attentions (torch.Tensor): The attention matrices of each layer.
            Shape: (n_layers, batch_size, n_heads, seq_len, seq_len)
        head_dim (int, optional): The dimension of the heads. Defaults to 2.

    Returns:
        torch.Tensor: The residual attention matrices of each layer.
            Shape: (n_layers, batch_size, seq_len, seq_len)
    """
    n_heads = attentions.shape[head_dim]

    attention_residuals = attentions.sum(axis=head_dim) / n_heads
    attention_residuals += torch.eye(
        attention_residuals.shape[head_dim], device=attentions.device
    )[None, ...]
    return attention_residuals / attention_residuals.sum(axis=-1)[..., None]


def compute_rollout_attention(
    attentions: torch.Tensor, add_residual: bool = False
) -> torch.Tensor:
    """Compute the joint attention matrix from the attention matrices of each layer.

    Args:
        attentions (torch.Tensor): The attention matrices of each layer.
            Shape: (n_layers, batch_size, (n_heads), seq_len, seq_len)
        add_residual (bool, optional): Whether to add a residuals.
            Defaults to False.

    Returns:
        torch.Tensor: The joint attention matrix.
            Shape: (n_layers, batch_size, seq_len, seq_len)
    """
    if add_residual:
        attention_residuals = torch.eye(attentions.shape[1])[None, ...]
        attentions += attention_residuals
        attentions /= attentions.sum(-1)[..., None]

    joint_attentions = torch.zeros(attentions.shape, device=attentions.device)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = attentions[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = attentions[i].matmul(joint_attentions[i - 1])

    return joint_attentions
