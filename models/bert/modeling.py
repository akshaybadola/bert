from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Final
import sys
import json
import math
import copy
import os
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from bert.modules import BertLayer, LinearActivation, BertEmbeddings, BertPooler
from bert.heads import BertPreTrainingHeads, BertOnlyMLMHead


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 output_all_encoded_layers=False):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if (isinstance(vocab_size_or_config_json_file, str)
                or (sys.version_info[0] == 2 and
                    isinstance(vocab_size_or_config_json_file, unicode))):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.output_all_encoded_layers = output_all_encoded_layers
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.output_all_encoded_layers = config.output_all_encoded_layers
        self._checkpoint_activations = False

    @torch.jit.unused
    def checkpointed_forward(self, hidden_states, attention_mask):
        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_
            return custom_forward

        layer_ind = 0
        num_layers = len(self.layer)
        chunk_length = math.ceil(math.sqrt(num_layers))
        while layer_ind < num_layers:
            hidden_states = checkpoint.checkpoint(custom(layer_ind, layer_ind+chunk_length),
                                                  hidden_states, attention_mask*1)
            layer_ind += chunk_length

        return hidden_states

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []

        if self._checkpoint_activations:
            hidden_states = self.checkpointed_forward(hidden_states, attention_mask)
        else:
            # (bsz, seq, hidden) => (seq, bsz, hidden)
            hidden_states = hidden_states.transpose(0, 1)
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)

                if self.output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            # The hidden states need to be contiguous at this point to enable
            # dense_sequence_output
            # (seq, bsz, hidden) => (bsz, seq, hidden)
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        if not self.output_all_encoded_layers or self._checkpoint_activations:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @torch.jit.ignore
    def checkpoint_activations(self, val):
        def _apply_flag(module):
            if hasattr(module, "_checkpoint_activations"):
                module._checkpoint_activations = val
        self.apply(_apply_flag)

    def enable_apex(self, val):
        def _apply_flag(module):
            if hasattr(module, "apex_enabled"):
                module.apex_enabled = val
        self.apply(_apply_flag)

    @classmethod
    def from_scratch(cls, pretrained_model_name_or_path, distill_config=None,
                     pooler=True, *inputs, **kwargs):
        resolved_config_file = os.path.join(
            pretrained_model_name_or_path, kwargs["config_name"])
        config = BertConfig.from_json_file(resolved_config_file)

        # Load distillation specific config
        if distill_config:
            distill_config = json.load(open(distill_config, "r"))
            distill_config["distillation_config"]["use_pooler"] = pooler
            config.__dict__.update(distill_config)

        print("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        return model, config

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
    #                     from_tf=False, distill_config=None, pooler=True, *inputs, **kwargs):
    #     """
    #     Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
    #     Download and cache the pre-trained model file if needed.

    #     Params:
    #         pretrained_model_name_or_path: either:
    #             - a str with the name of a pre-trained model to load selected in the list of:
    #                 . `bert-base-uncased`
    #                 . `bert-large-uncased`
    #                 . `bert-base-cased`
    #                 . `bert-large-cased`
    #                 . `bert-base-multilingual-uncased`
    #                 . `bert-base-multilingual-cased`
    #                 . `bert-base-chinese`
    #             - a path or url to a pretrained model archive containing:
    #                 . `bert_config.json` a configuration file for the model
    #                 . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
    #             - a path or url to a pretrained model archive containing:
    #                 . `bert_config.json` a configuration file for the model
    #                 . `model.chkpt` a TensorFlow checkpoint
    #         from_tf: should we load the weights from a locally saved TensorFlow checkpoint
    #         cache_dir: an optional path to a folder in which the pre-trained models will be cached.
    #         state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
    #         *inputs, **kwargs: additional input for the specific Bert class
    #             (ex: num_labels for BertForSequenceClassification)
    #     """
    #     if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
    #         archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
    #     else:
    #         archive_file = pretrained_model_name_or_path
    #     # redirect to the cache, if necessary
    #     try:
    #         resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
    #     except EnvironmentError:
    #         logger.error(
    #             "Model name '{}' was not found in model name list ({}). "
    #             "We assumed '{}' was a path or url but couldn't find any file "
    #             "associated to this path or url.".format(
    #                 pretrained_model_name_or_path,
    #                 ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
    #                 archive_file))
    #         return None
    #     if resolved_archive_file == archive_file:
    #         logger.info("loading archive file {}".format(archive_file))
    #     else:
    #         logger.info("loading archive file {} from cache at {}".format(
    #             archive_file, resolved_archive_file))
    #     tempdir = None
    #     if os.path.isdir(resolved_archive_file) or from_tf:
    #         serialization_dir = resolved_archive_file
    #     else:
    #         # Extract archive to temp dir
    #         tempdir = tempfile.mkdtemp()
    #         logger.info("extracting archive file {} to temp dir {}".format(
    #             resolved_archive_file, tempdir))
    #         with tarfile.open(resolved_archive_file, 'r:gz') as archive:
    #             archive.extractall(tempdir)
    #         serialization_dir = tempdir
    #     # Load config
    #     config_file = os.path.join(serialization_dir, CONFIG_NAME)
    #     config = BertConfig.from_json_file(config_file)
    #     # Load distillation specific config
    #     if distill_config:
    #         distill_config = json.load(open(distill_config, "r"))
    #         distill_config["distillation_config"]["use_pooler"] = pooler
    #         config.__dict__.update(distill_config)

    #     logger.info("Model config {}".format(config))
    #     # Instantiate model.
    #     model = cls(config, *inputs, **kwargs)
    #     if state_dict is None and not from_tf:
    #         weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
    #         state_dict = torch.load(weights_path, map_location='cpu')
    #     if tempdir:
    #         # Clean up temp dir
    #         shutil.rmtree(tempdir)
    #     if from_tf:
    #         # Directly load from a TensorFlow checkpoint
    #         weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
    #         return load_tf_weights_in_bert(model, weights_path)
    #     # Load from a PyTorch state_dict
    #     old_keys = []
    #     new_keys = []
    #     for key in state_dict.keys():
    #         new_key = None
    #         if 'gamma' in key:
    #             new_key = key.replace('gamma', 'weight')
    #         if 'beta' in key:
    #             new_key = key.replace('beta', 'bias')
    #         if 'intermediate.dense.' in key:
    #             new_key = key.replace('intermediate.dense.', 'intermediate.dense_act.')
    #         if 'pooler.dense.' in key:
    #             new_key = key.replace('pooler.dense.', 'pooler.dense_act.')
    #         if new_key:
    #             old_keys.append(key)
    #             new_keys.append(new_key)
    #     for old_key, new_key in zip(old_keys, new_keys):
    #         state_dict[new_key] = state_dict.pop(old_key)

    #     missing_keys = []
    #     unexpected_keys = []
    #     error_msgs = []
    #     # copy state_dict so _load_from_state_dict can modify it
    #     metadata = getattr(state_dict, '_metadata', None)
    #     state_dict = state_dict.copy()
    #     if metadata is not None:
    #         state_dict._metadata = metadata

    #     def load(module, prefix=''):
    #         local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    #         module._load_from_state_dict(
    #             state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
    #         for name, child in module._modules.items():
    #             if child is not None:
    #                 load(child, prefix + name + '.')
    #     start_prefix = ''
    #     if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
    #         start_prefix = 'bert.'
    #     load(model, prefix=start_prefix)
    #     if len(missing_keys) > 0:
    #         logger.info("Weights of {} not initialized from pretrained model: {}".format(
    #             model.__class__.__name__, missing_keys))
    #     if len(unexpected_keys) > 0:
    #         logger.info("Weights from pretrained model not used in {}: {}".format(
    #             model.__class__.__name__, unexpected_keys))
    #     if len(error_msgs) > 0:
    #         raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    #                            model.__class__.__name__, "\n\t".join(error_msgs)))
    #     return model, config


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    distillation: Final[bool]
    teacher: Final[bool]

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        # Distillation specific
        self.distillation = getattr(config, 'distillation', False)
        if self.distillation:
            self.distill_state_dict = OrderedDict()
            self.distill_config = config.distillation_config
        else:
            self.distill_config = {'use_pooler': False, 'use_pred_states': False}

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        # Use pooler if not running distillation or distill_config["use_pooler"] is set to True
        if (not self.distillation or
            (self.distill_config["use_pooler"] and
             self.distill_config["use_pred_states"])):
            self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        self.output_all_encoded_layers = config.output_all_encoded_layers
        self.teacher = False

    def forward(self, input_ids, token_type_ids, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.embeddings.word_embeddings.weight.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = encoded_layers[-1]
        # Use pooler if not running distillation or distill_config["use_pooler"] is set to True
        if (not self.distillation or
            (self.distill_config["use_pooler"] and
             self.distill_config["use_pred_states"])):
            pooled_output = self.pooler(sequence_output)
        else:
            pooled_output = None
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1:]
        if not self.teacher:
            return encoded_layers, pooled_output

    def make_teacher(self, ):
        self.teacher = True


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    distillation: Final[bool]

    def __init__(self, config, sequence_output_is_dense=False):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.distillation = getattr(config, 'distillation', False)
        if not self.distillation:
            self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight,
                                            sequence_output_is_dense)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels):
        # if self.distillation:
        #     self.bert(input_ids, token_type_ids, attention_mask)
        # else:
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        if not self.distillation:
            sequence_output = encoded_layers[-1]
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output,
                                                                 masked_lm_labels)
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = encoded_layers[-1]
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                      masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores
