from typing import Final
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..functions import ACT2FN


class LinearActivation(nn.Module):
    r"""Fused Linear and activation Module.
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, act='gelu', bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None
        assert act in ACT2FN, "Activation function is not found in activation dictionary."
        self.act_fn = ACT2FN[act]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # if not self.bias is None:
        #     return self.biased_act_fn(self.bias, F.linear(input, self.weight, None))
        # else:
        return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    distillation: Final[bool]

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.distillation = getattr(config, 'distillation', False)
        if self.distillation:
            self.distill_state_dict = OrderedDict()
            self.distill_config = config.distillation_config
        else:
            self.distill_config = {'use_embedding_states': False}

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        if self.distillation:
            if self.distill_config["use_embedding_states"]:
                self.distill_state_dict["embedding_states"] = embeddings
        return embeddings


class BertSelfAttention(nn.Module):
    distillation: Final[bool]

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Distillation specific
        self.distillation = getattr(config, 'distillation', False)
        if self.distillation:
            self.distill_state_dict = OrderedDict()
            self.distill_config = config.distillation_config
        else:
            self.distill_config = {'use_attention_scores': False, 'use_value_states': False}

    def transpose_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(0)
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads,
                   self.attention_head_size).transpose(0, 1)
        return x

    def transpose_key_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(0)
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads,
                   self.attention_head_size).permute(1, 2, 0)
        return x

    def forward(self, hidden_states, attention_mask):
        # (seq, bsz, hidden)
        batch_size = hidden_states.size(1)
        seq_length = hidden_states.size(0)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.bmm(query_layer, key_layer)
        # (bsz, heads, seq, seq)
        attention_scores = attention_scores.view(batch_size,
                                                 self.num_attention_heads,
                                                 seq_length, seq_length)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # (bsz, heads, seq, seq)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.view(batch_size * self.num_attention_heads,
                                               seq_length, seq_length)

        context_layer = torch.bmm(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 1).contiguous()
        # (seq, bsz, hidden)
        context_layer = context_layer.view(seq_length, batch_size, self.all_head_size)

        # Cache states if running distillation
        if self.distillation:
            if self.distill_config["use_attention_scores"]:
                self.distill_state_dict["attention_scores"] = attention_scores
            if self.distill_config["use_value_states"]:
                self.distill_state_dict["value_states"] = context_layer
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size,
                                          config.intermediate_size, act=config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    distillation: Final[bool]

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        # Distillation specific
        self.distillation = getattr(config, 'distillation', False)
        if self.distillation:
            self.distill_state_dict = OrderedDict()
            self.distill_config = config.distillation_config
        else:
            self.distill_config = {'use_hidden_states': False}

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        # Cache states if running distillation
        if self.distillation:
            if self.distill_config["use_hidden_states"]:
                self.distill_state_dict["hidden_states"] = layer_output
        return layer_output


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size, config.hidden_size, act="tanh")

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output    
