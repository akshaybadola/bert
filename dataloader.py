import io
from functools import partial

import torch
import numpy as np

from transformers import AutoTokenizer
import datasets


# TODO: Add seed for numpy maybe
class BertDataset(torch.utils.data.Dataset):
    """Return a sentence pair from the dataset

    Pick two sentences :code:`sentence_a`, and :code:`sentence_b` from data

    50% of the time :code:`sentence_b` is next sentence, rest random

    Return [CLS] :code:`sentence_a` [SEP] :code:`sentence_b` [SEP]
    and :code:`next_sent` if next sentence is :code:`sentence_b`

    Args:
        location: Path to dataset
        shuffle: Shuffle the dataset after loading
        in_memory: Load the entire dataset in memory

    :code:`in_memory` doesn't seem to lead to any significant speedup and can be avoided
    though it can be tested individually

    """

    def __init__(self, location, shuffle=True, in_memory=False):
        self.data = datasets.load_from_disk(location, keep_in_memory=in_memory)
        self.inds = np.arange(len(self.data))
        if shuffle:
            print("Shuffling dataset")
            np.random.shuffle(self.inds)

    def reset(self):
        np.random.shuffle(self.inds)

    def __getitem__(self, i):
        output = {x: None for x in
                  ["input_ids", "token_type_ids",
                   "attention_mask", "special_tokens_mask",
                   "next_sentence_label"]}
        if np.random.choice([0, 1]):
            sent_a, sent_b, next_sent =\
                self.data[int(self.inds[i])], self.data[int(self.inds[i+1])], 1
        else:
            sent_a, sent_b, next_sent =\
                self.data[int(self.inds[i])], self.data[int(np.random.choice(self.inds))], 0
        len_a = len(sent_a['input_ids'])
        len_b = len(sent_b['input_ids'])
        # Since the sentences in the dataset are truncated to 512, if we
        # concatenate two sentences it becomes > 512. In that case, reduce
        # the lengths of each proportionally
        if len_a + len_b > 512:
            limit_a = int(512 * len_a / (len_a+len_b))+1
            limit_b = int(512 * len_b / (len_a+len_b))
        else:
            limit_a = len_a
            limit_b = len_b
        sep = sent_a['input_ids'][-1]
        output["input_ids"] = torch.as_tensor([*sent_a['input_ids'][:limit_a-1], sep,
                                               *sent_b['input_ids'][1:limit_b-1], sep],
                                              dtype=torch.long)
        output["token_type_ids"] = torch.as_tensor([*sent_a['token_type_ids'][:limit_a],
                                                    *[1 for _ in sent_b['token_type_ids'][1:limit_b]]],
                                                   dtype=torch.long)
        output["attention_mask"] = torch.ones(len(sent_a['input_ids'][:limit_a])
                                              + len(sent_b['input_ids'][:limit_b]) - 1,
                                              dtype=torch.long)
        output["special_tokens_mask"] = torch.as_tensor([*sent_a['special_tokens_mask'][:limit_a-1], 1,
                                                         *sent_b['special_tokens_mask'][1:limit_b-1], 1],
                                                        dtype=torch.long)
        output["next_sentence_label"] = next_sent
        return output

    def __len__(self):
        return len(self.data)


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
    """Prepare masked tokens inputs/labels for masked language modeling
    80% MASK, 10% random, 10% original.

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
    # static_masking = (len(batch[0]) > 3)
    static_masking = (len(batch.keys()) > 3)
    if static_masking:
        # assert len(batch[0]) == 5
        assert len(batch.keys()) == 5
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


def collator_alt(batch, seq_align_len, tokenizer):
    output = {x: None for x in ["input_ids", "token_type_ids", "attention_mask",
                                "masked_lm_labels", "next_sentence_label"]}
    with torch.no_grad():
        inputs = batch['input_ids']
        batch_size = len(inputs)
        lengths = [*map(len, inputs)]
        max_len = max(lengths)
        width = int(np.ceil(max_len / seq_align_len) * seq_align_len)
        size = (batch_size, width)
        input_tensor = torch.zeros(size, dtype=torch.long)
        mask_tensor = torch.zeros(size, dtype=torch.long)
        for i, x in enumerate(inputs):
            input_tensor[i, :lengths[i]] = torch.tensor(x)
            # CHECK: Do we mask out CLS, SEP?
            mask_tensor[i, :lengths[i]] = 1
        output["input_ids"], output["masked_lm_labels"] = _mask_tokens(input_tensor, tokenizer=tokenizer)
        output["attention_mask"] = mask_tensor
    return output


def collator(batch, seq_align_len, tokenizer, pad_full=False):
    output = {x: None for x in ["input_ids", "token_type_ids", "attention_mask",
                                "masked_lm_labels", "next_sentence_labels"]}
    lengths = [len(x['input_ids']) for x in batch]
    batch_size = len(batch)
    max_len = max(lengths)
    width = 512 if pad_full else int(np.ceil(max_len / seq_align_len) * seq_align_len)
    size = (batch_size, width)
    input_tensor = torch.zeros(size, dtype=torch.long)
    mask_tensor = torch.zeros(size, dtype=torch.long)
    special_tokens_mask_tensor = torch.zeros(size, dtype=torch.long)
    token_type_id_tensor = torch.zeros(size, dtype=torch.long)
    next_sentence_labels = []
    with torch.no_grad():
        for i, x in enumerate(batch):
            input_tensor[i, :lengths[i]] = x['input_ids']
            # CHECK: Do we mask out CLS, SEP?
            mask_tensor[i, :lengths[i]] = x['attention_mask']
            special_tokens_mask_tensor[i, :lengths[i]] = x['special_tokens_mask']
            token_type_id_tensor[i, :lengths[i]] = x['token_type_ids']
            next_sentence_labels.append(x['next_sentence_label'])
        output["input_ids"], output["masked_lm_labels"] = _mask_tokens(
            input_tensor, tokenizer=tokenizer,
            special_tokens_mask=special_tokens_mask_tensor)
        output["attention_mask"] = mask_tensor
        output["token_type_ids"] = token_type_id_tensor
        output["next_sentence_labels"] = torch.as_tensor(next_sentence_labels)
    return output



def get_wiki_books_loader(batch_size, num_workers, seq_align_len, shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased-whole")
    collate_fn = partial(collator, seq_align_len=seq_align_len, tokenizer=tokenizer)
    data = BertDataset("books-wiki-tokenized", shuffle=shuffle)
    loader = torch.utils.data.DataLoader(data, shuffle=False,
                                         num_workers=num_workers, drop_last=False,
                                         pin_memory=True, batch_size=batch_size,
                                         collate_fn=collate_fn)
    return loader
