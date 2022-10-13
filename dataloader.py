import io

import torch
import numpy as np


def serialize_np_array(a):
    memfile = io.BytesIO()
    np.save(memfile, a)
    memfile.seek(0)
    return memfile.read()


def deserialize_np_array(b):
    memfile = io.BytesIO()
    memfile.write(b)
    memfile.seek(0)
    return np.load(memfile)


def _mask_tokens(inputs, special_tokens_mask=None, tokenizer=None,
                 mlm_probability=0.15, ignore_index=-1):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK,
    10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability
    # `mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    # We only compute loss on masked tokens
    labels[~masked_indices] = ignore_index

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token
    # ([MASK])
    indices_replaced = (torch.bernoulli(torch.full(labels.shape, 0.8)).bool() &
                        masked_indices)
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool() &
                      masked_indices & ~indices_replaced)
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens
    # unchanged
    return inputs, labels


def _to_encoded_inputs(batch, tokenizer, sequence_length_alignment=8,
                       ignore_index=-1):
    batch_size = len(batch)
    As, Bs, are_random_next = [], [], []
    static_masking = (len(batch[0]) > 3)
    if static_masking:
        assert len(batch[0]) == 5
        all_masked_lm_positions, all_masked_lm_labels = [], []
    # Unpack each field.
    for sample in batch:
        As.append(tuple(sample[0].split()))
        Bs.append(tuple(sample[1].split()))
        are_random_next.append(sample[2])
        if static_masking:
            all_masked_lm_positions.append(
                torch.from_numpy(deserialize_np_array(sample[3]).astype(int)))
            all_masked_lm_labels.append(sample[4].split())
    # Figure out the sequence length of this batch.
    batch_seq_len = max(
        (len(tokens_A) + len(tokens_B) + 3 for tokens_A, tokens_B in zip(As, Bs)))
    # Align the batch_seq_len to a multiple of sequence_length_alignment, because
    # TC doesn't like it otherwise.
    batch_seq_len = (((batch_seq_len - 1) // sequence_length_alignment + 1) *
                     sequence_length_alignment)
    # Allocate the input torch.Tensor's.
    input_ids = torch.zeros(batch_size, batch_seq_len, dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = torch.zeros_like(input_ids)
    if static_masking:
        labels = torch.full_like(input_ids, ignore_index)
    else:
        special_tokens_mask = torch.zeros_like(input_ids)
    # Fill in the input torch.Tensor's.
    for sample_idx in range(batch_size):
        tokens_A, tokens_B = As[sample_idx], Bs[sample_idx]
        # Prepare the input token IDs.
        tokens = ('[CLS]',) + tokens_A + ('[SEP]',) + tokens_B + ('[SEP]',)
        input_ids[sample_idx, :len(tokens)] = torch.as_tensor(
            tokenizer.convert_tokens_to_ids(tokens),
            dtype=torch.long,
        )
        # Prepare the token type ids (segment ids).
        start_idx = len(tokens_A) + 2
        end_idx = len(tokens_A) + len(tokens_B) + 3
        token_type_ids[sample_idx, start_idx:end_idx] = 1
        # Prepare the attention mask (input mask).
        attention_mask[sample_idx, :end_idx] = 1
        if static_masking:
            # Prepare the MLM labels.
            labels[sample_idx, all_masked_lm_positions[sample_idx]] = torch.as_tensor(
                tokenizer.convert_tokens_to_ids(all_masked_lm_labels[sample_idx]),
                dtype=torch.long,
            )
        else:
            # Prepare special_tokens_mask (for DataCollatorForLanguageModeling)
            special_tokens_mask[sample_idx, 0] = 1
            special_tokens_mask[sample_idx, len(tokens_A) + 1] = 1
            special_tokens_mask[sample_idx, len(tokens_A) + len(tokens_B) + 2:] = 1
    # Compose output dict.
    encoded_inputs = {'input_ids': input_ids,
                      'token_type_ids': token_type_ids,
                      'attention_mask': attention_mask,
                      'next_sentence_labels': torch.as_tensor(
                          are_random_next,
                          dtype=torch.long,
                      )}
    if static_masking:
        encoded_inputs['labels'] = labels
    else:
        encoded_inputs['special_tokens_mask'] = special_tokens_mask
    return encoded_inputs
