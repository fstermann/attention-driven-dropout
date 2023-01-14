from __future__ import annotations

import logging
from typing import Callable

import numpy as np
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
            dynamically from the sequence length (seq_length // min_tokens). Defaults to False.
        summation (str, optional): The method to use to sum up the attention scores. Defaults to "naive".
            - "naive": Sum up the attention scores for each token, each batch (over the layers and heads)
            - "flow": Use the Attention Flow method to sum up the attention scores
            - "rollout": Use the Attention Rollout method to sum up the attention scores

    """

    def __init__(
        self,
        model: str | BertModel | RobertaModel,
        *,
        n_dropout: int = 1,
        min_tokens: int = 10,
        dynamic_dropout: bool = False,
        summation_method: str = "naive",
    ):
        super().__init__()
        self.n_dropout = n_dropout
        self.min_tokens = min_tokens
        self.dynamic_dropout = dynamic_dropout

        self.model = self.get_model(model)
        logging.debug("[ADD] model: ", self.model.base_model_prefix)      

        self.padding_token = self.get_padding_token(model)
        logging.debug("[ADD] padding_token: ", self.padding_token)

        self.summation_method = summation_method
        self.summation_func = self.get_summation_func(summation_method)
        logging.debug("[ADD] summation_method: ", self.summation_method)
        logging.debug("[ADD] summation_func: ", self.summation_func)


    def get_model(self, model: str | BertModel | RobertaModel) -> BertModel | RobertaModel:
        if not isinstance(model, str):
            return model
        if model == "bert-base-uncased":
            return BertModel.from_pretrained(model)
        if model == "roberta-base":
            return RobertaModel.from_pretrained(model)
        raise ValueError(f"Model {model} is not supported.")

    
    def get_padding_token(self, model: BertModel | RobertaModel) -> int:
        if model.base_model_prefix == "bert":
            # Using BERT tokenizer
            return 0
        if model.base_model_prefix == "roberta":
            # Using RoBERTa tokenizer
            return 1
        raise ValueError(f"Model {model.base_model_prefix} is not supported.")

    def get_summation_func(self, summation_method: str) -> Callable:
        SUMMATION_CHOICES = {
            "naive": self._sum_attentions_naive,
            "flow": self._sum_attentions_flow,
            "rollout": self._sum_attentions_rollout,
        }
        if summation_method not in SUMMATION_CHOICES:
            raise ValueError(f"Summation method {summation_method} is not supported.")
        
        return SUMMATION_CHOICES[summation_method]

    def forward(
        self, input_ids: torch.Tensor, return_scores: bool = False
    ) -> torch.Tensor:
        if not self.training:
            return input_ids

        # Use the pre-trained attention model to calculate the attention probabilities
        mask = input_ids != self.padding_token
        output: BaseModelOutputWithPoolingAndCrossAttentions = self.model(
            input_ids,
            attention_mask=mask,
            output_attentions=True,
        )
        # output.attentions.shape: (n_layers, batch_size, num_heads, sequence_length, sequence_length)

        attentions = torch.stack(output.attentions)
        _, batch_size, _, _, seq_length = attentions.shape

        n_dropout = (
            seq_length // self.min_tokens if self.dynamic_dropout else self.n_dropout
        )

        attention_sums = self.summation_func(
            attentions
        )  # Sum over layers, heads and sequence length

        min_indices = attention_sums.topk(n_dropout, dim=1, largest=False).indices

        # Drop min index in every second sentence
        input_ids = input_ids.clone()
        for i in range(1, batch_size, 2):
            sequence = input_ids[i].clone()
            n_tokens = len(sequence[sequence != self.padding_token])
            ind = min_indices[i]

            if self.dynamic_dropout:
                n_dropout = n_tokens // self.min_tokens
                ind = ind[:n_dropout]

                if len(ind) == 0:
                    continue
            else:
                # Only replace words if there are more than min_tokens words in the sequence
                if n_tokens < self.min_tokens:
                    continue

            sequence[ind] = self.padding_token
            sequence_tokens = sequence[
                sequence != self.padding_token
            ]  # Without padding tokens

            # Shift tokens to the left
            ind = torch.arange(0, len(sequence_tokens), device=input_ids.device)
            result = torch.full(
                sequence.shape, self.padding_token, device=input_ids.device
            )
            input_ids[i] = result.scatter(0, ind, sequence_tokens)

        if return_scores:
            return input_ids.detach(), attention_sums

        return input_ids.detach()

    

    def _sum_attentions_naive(
        self, attentions: torch.Tensor, dim: tuple[int, ...] = (0, 2, 3)
    ) -> torch.Tensor:
        """Compute the summed attention scores for each token in the sequence.

        Args:
            attentions (torch.Tensor): The attention scores of shape (n_layers, batch_size, num_heads, seq_len, seq_len)

        Returns:
            torch.Tensor: The summed attention scores of shape (batch_size, seq_len)
        """
        attentions[
            attentions == self.padding_token
        ] = 1  # Replace masked attention with 1, to avoid 0 in min
        return torch.sum(
            attentions, dim=dim
        )  # Sum over layers, heads and sequence length

    def _sum_attentions_flow(
        self, attentions: torch.Tensor, dim: tuple[int, ...] = (1, 2)
    ) -> torch.Tensor:
        """Compute the summed attention scores for each token in the sequence with the Attention Flow method.

        Args:
            attentions (torch.Tensor): The attention scores of shape (n_layers, batch_size, num_heads, seq_len, seq_len)

        Returns:
            torch.Tensor: The summed attention scores of shape (batch_size, seq_len)
        """
        raise NotImplementedError

    def _sum_attentions_rollout(
        self, attentions: torch.Tensor, dim: tuple[int, ...] = (0, 2)
    ) -> torch.Tensor:
        """Compute the summed attention scores for each token in the sequence with the Attention Rollout method.

        Args:
            attentions (torch.Tensor): The attention scores of shape (n_layers, batch_size, num_heads, seq_len, seq_len)

        Returns:
            torch.Tensor: The summed attention scores of shape (batch_size, seq_len)
        """
        residual_attentions = get_residual_attentions(attentions)
        rollout_attentions = compute_rollout_attention(
            residual_attentions, add_residual=False
        )

        # rollout_attentions  shape (n_layers, batch_size, seq_len, seq_len)
        # output =  rollout_attentions.sum(dim=dim)
        # output = rollout_attentions[-1].sum(dim=dim) # Take last layer
        last_layer = rollout_attentions[-1]
        output = last_layer.sum(dim=((1)))  # Take last
        # output = rollout_attentions[:,-1,:,:]
        # Need to replace masked attentions with inf, to be excluded from min
        # -> Take the first layer, first head, for each batch element
        mask = attentions[0, :, 1, 1] != self.padding_token
        output[~mask] = float("inf")  # TODO: Is there a better way?
        return output


class RandomDropout(nn.Module):
    def __init__(
        self, n_dropout: int = 1, min_tokens: int = 10, dynamic_dropout: bool = False
    ) -> None:
        super().__init__()
        self.n_dropout = n_dropout
        self.min_tokens = min_tokens
        self.dynamic_dropout = dynamic_dropout

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return input_ids

        batch_size = input_ids.shape[0]
        padding_token = self.get_padding_token(input_ids)

        input_ids = input_ids.clone()
        for i in range(1, batch_size, 2):
            sequence = input_ids[i].clone()
            n_tokens = len(sequence[sequence > padding_token])
            n_dropout = (
                n_tokens // self.min_tokens if self.dynamic_dropout else self.n_dropout
            )

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

            sequence[ind] = padding_token
            sequence_tokens = sequence[
                sequence != padding_token
            ]  # Without padding tokens

            # Shift tokens to the left
            ind = torch.arange(0, len(sequence_tokens), device=input_ids.device)
            result = torch.full(sequence.shape, padding_token, device=input_ids.device)
            input_ids[i] = result.scatter(0, ind, sequence_tokens)

        return input_ids.detach()

    def get_padding_token(self, input_ids) -> int:
        if input_ids[0][0] == 0:
            # We are using BERT tokenizer
            return 0
        if input_ids[0][0] == 1:
            # We are using RoBERTa tokenizer
            return 1

        raise ValueError(f"Unkown padding token for input_ids: {input_ids[0][0]}")


## ATTENTION FLOW/ROLLOUT FUNCTIONS


def get_residual_attentions(raw_attentions: torch.Tensor, head_dim: int = 2):
    # TODO: What does [None, ...] do?
    res_att_mat = raw_attentions.sum(axis=head_dim) / raw_attentions.shape[head_dim]
    res_att_mat = (
        res_att_mat
        + torch.eye(res_att_mat.shape[head_dim], device=raw_attentions.device)[
            None, ...
        ]
    )
    res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[..., None]

    return res_att_mat


# joint_attentions = compute_joint_attention(res_att_mat, add_residual=False)


def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = torch.zeros(
        ((n_layers + 1) * length, (n_layers + 1) * length), device=mat.device
    )
    labels_to_index = {}
    for k in np.arange(length):
        labels_to_index[str(k) + "_" + input_tokens[k]] = k

    for i in np.arange(1, n_layers + 1):
        for k_f in np.arange(length):
            index_from = (i) * length + k_f
            label = "L" + str(i) + "_" + str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i - 1) * length + k_t
                adj_mat[index_from][index_to] = mat[i - 1][k_f][k_t]

    return adj_mat, labels_to_index


def compute_rollout_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = torch.eye(att_mat.shape[1])[None, ...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = att_mat

    joint_attentions = torch.zeros(aug_att_mat.shape, device=aug_att_mat.device)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1, layers):
        # joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1]).reshape((joint_attentions[i].shape))
        # joint_attentions[i] = torch.einsum('ijk,ijk->ij', aug_att_mat[i], joint_attentions[i-1]).unsqueeze(-1)
        joint_attentions[i] = aug_att_mat[i].matmul(joint_attentions[i - 1])

    return joint_attentions
