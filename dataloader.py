import io
from functools import partial

import torch
import numpy as np

from transformers import AutoTokenizer
import datasets

from common_pyutil.monitor import Timer
timer = Timer()


def get_limits(len_a, len_b, max_seq_len, min_seq_len, truncate_strategy):
    # Since the max sentence length in the dataset is 512, if we
    # concatenate two sentences it becomes > 512. In that case, reduce
    # the lengths according to a strategy
    if len_a + len_b > max_seq_len:
        if truncate_strategy == "proportional":
            limit_a = int(max_seq_len * len_a / (len_a+len_b))
            limit_b = int(max_seq_len * len_b / (len_a+len_b))
        elif truncate_strategy == "truncate_second":
            min_len_b = min_seq_len
            max_len_a = max_seq_len - min(min_len_b, len_b)
            if len_a > max_len_a:
                limit_a = max_len_a
                limit_b = len_b if len_b < min_len_b else min_len_b
            else:
                limit_a = len_a
                limit_b = max_seq_len - len_a
        else:
            raise ValueError(f"Unknown truncate strategy {self.truncate_strategy}")
    else:
        limit_a = len_a
        limit_b = len_b
    return limit_a, limit_b


class BertDatasetWithSense(torch.utils.data.Dataset):
    # send
    def __getitem__(self, i):
        {"important_word_mask": torch.as_tensor([0])}
        pass


class BertDatasetOWT(torch.utils.data.Dataset):
    """Return a sentence pair from the dataset

    Pick one `document` from dataset. Then pick one sentence :code:`sentence_a`
    and half the time :code:`sentence_b` from same `document` and other half
    from another.

    Return [CLS] :code:`sentence_a` [SEP] :code:`sentence_b` [SEP]
    and :code:`next_sent` if :code:`sentence_b` if from same document

    Args:
        location: Path to dataset
        whole_word_mask: Mask whole words. Defaults to True
        max_seq_len: Maximum length of the training example
        min_seq_len: Minimum length of the training example. Only
                     used with :code:`truncate_second` truncate_strategy
        truncate_stragety: The strategy to use if the training example > max_seq_len
                           one of :code:`proportional` or :code:`truncate_second`
    :code:`in_memory` doesn't seem to lead to any significant speedup and can be avoided
    though it can be tested individually

    """

    def __init__(self, location, tokenizer, whole_word_mask=True, max_seq_len=128,
                 min_seq_len=30, truncate_strategy="truncate_second"):
        if not whole_word_mask:
            raise NotImplementedError("Only whole word masking is implemented right now")
        self.data = datasets.load_from_disk(location)
        self.tokenizer = tokenizer
        self.inds = np.arange(len(self.data))
        self.max_ind = len(self.inds)
        self.truncate_strategy = truncate_strategy
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id

    def __getitem__(self, i):
        output = {}
        with timer:
            doc_a = self.data[i]
            token_lines = doc_a['tokens']
            if not len(token_lines):
                while not len(token_lines):
                    ind_a = np.random.randint(self.max_ind)
                    doc_a = self.data[ind_a]
                    token_lines = doc_a['tokens']
            if len(token_lines) == 1:
                ind = int(np.random.randint(self.max_ind))
                ind_a = 0
                words_a = token_lines[ind_a]
                token_ids_a = doc_a['input_ids'][ind_a]
                doc_b = self.data[ind]
                if not len(doc_b['tokens']):
                    while not len(doc_b['tokens']):
                        ind = int(np.random.randint(self.max_ind))
                        doc_b = self.data[ind]
                ind_b = np.random.choice(len(doc_b['tokens']))
                words_b = doc_b['tokens'][ind_b]
                token_ids_b = doc_b['input_ids'][ind_b]
                next_sent = 0
            elif len(token_lines) > 1:
                if np.random.randint(2):
                    ind_a, ind_b = np.random.choice(len(token_lines), 2, replace=False)
                    words_a, words_b = token_lines[ind_a], token_lines[ind_b]
                    token_ids_a, token_ids_b = doc_a['input_ids'][ind_a], doc_a['input_ids'][ind_b]
                    next_sent = 1
                else:
                    ind = int(np.random.randint(self.max_ind))
                    doc_b = self.data[ind]
                    if not len(doc_b['tokens']):
                        while not len(doc_b['tokens']):
                            ind = int(np.random.randint(self.max_ind))
                            doc_b = self.data[ind]
                    ind_a = np.random.randint(len(token_lines))
                    words_a = token_lines[ind_a]
                    token_ids_a = doc_a['input_ids'][ind_a]
                    ind_b = np.random.randint(len(doc_b['tokens']))
                    words_b = doc_b['tokens'][ind_b]
                    token_ids_b = doc_b['input_ids'][ind_b]
                    next_sent = 0
        time_a = timer.time
        with timer:
            len_a = len(words_a)
            len_b = len(words_b)
            limit_a, limit_b = get_limits(len_a, len_b, self.max_seq_len - 3,
                                          self.min_seq_len, self.truncate_strategy)  # 3 special tokens
            output = {}
            split_a = []
            split_b = []
            i = 1
            j = 1
            for x in words_a[:limit_a]:
                if x.startswith("#"):
                    split_a.append(j)
                else:
                    split_a.append(i)
                    j = i
                    i += 1
            i = 1
            j = 1
            for x in words_b[:limit_b]:
                if x.startswith("#"):
                    split_b.append(j)
                else:
                    split_b.append(i)
                    j = i
                    i += 1
        time_b = timer.time
        with timer:
            output = {}
            output["input_ids"] = torch.as_tensor(
                [self.cls, *token_ids_a[:limit_a],
                 self.sep, *token_ids_b[:limit_b], self.sep],
                dtype=torch.long)
            output["token_type_ids"] = torch.cat((torch.zeros(limit_a+2, dtype=torch.long),
                                                  torch.ones(limit_b+1, dtype=torch.long)))
            output["attention_mask"] = torch.ones_like(output['input_ids'])
            output["special_tokens_mask"] = torch.as_tensor(
                self.tokenizer.get_special_tokens_mask(output['input_ids'],
                                                       already_has_special_tokens=True),
                dtype=torch.long)
            output['split_tokens'] = torch.as_tensor([0, *split_a, 0,
                                                      *(torch.as_tensor(split_b) + max(split_a)), 0])
            output['split_range'] = torch.arange(1, output['split_tokens'].max()+1)
            output["next_sentence_label"] = next_sent
        time_c = timer.time            
        return output           #, time_a, time_b, time_c

    def __len__(self):
        return len(self.data)


# TODO: Add seed for numpy maybe
class BertDatasetBooksWiki(torch.utils.data.Dataset):
    """Return a sentence pair from the dataset

    Pick two sentences :code:`sentence_a`, and :code:`sentence_b` from data

    50% of the time :code:`sentence_b` is next sentence, rest random

    Return [CLS] :code:`sentence_a` [SEP] :code:`sentence_b` [SEP]
    and :code:`next_sent` if next sentence is :code:`sentence_b`

    Args:
        location: Path to dataset
        shuffle: Shuffle the dataset after loading
        in_memory: Load the entire dataset in memory
        max_seq_len: Maximum length of the training example
        min_seq_len: Minimum length of the training example. Only
                     used with :code:`truncate_second` truncate_strategy
        truncate_stragety: The strategy to use if the training example > max_seq_len
                           one of :code:`proportional` or :code:`truncate_second`
    :code:`in_memory` doesn't seem to lead to any significant speedup and can be avoided
    though it can be tested individually

    """

    def __init__(self, location, tokenizer, shuffle=True,
                 whole_word_mask=True, max_seq_len=512,
                 min_seq_len=None, truncate_strategy="proportional"):
        self.data = datasets.load_from_disk(location)
        self.tokenizer = tokenizer
        self.inds = np.arange(len(self.data))
        self.truncate_strategy = truncate_strategy
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        if shuffle:
            print("Shuffling dataset")
            np.random.shuffle(self.inds)

    def reset(self):
        np.random.shuffle(self.inds)

    def __getitem__(self, i):
        output = {}
        with timer:
            if np.random.choice([0, 1]):
                sent_a, sent_b, next_sent =\
                    self.data[int(self.inds[i])], self.data[int(self.inds[i+1])], 1
            else:
                sent_a, sent_b, next_sent =\
                    self.data[int(self.inds[i])], self.data[int(np.random.choice(self.inds))], 0
            len_a = len(sent_a['input_ids'])
            len_b = len(sent_b['input_ids'])
        time_a = timer.time
        with timer:
            limit_a, limit_b = get_limits(len_a, len_b, self.max_seq_len - 3,
                                          self.min_seq_len, self.truncate_strategy)  # 3 special tokens
            if "tokens" in sent_a:
                words_a = sent_a['tokens'][:limit_a]
                words_b = sent_b['tokens'][:limit_b]
            else:
                words_a = self.tokenizer.tokenize(sent_a['text'])[:limit_a]
                words_b = self.tokenizer.tokenize(sent_b['text'])[:limit_b]
            output = {}
            split_a = []
            split_b = []
            i = 1
            j = 1
            for x in words_a:
                if x.startswith("#"):
                    split_a.append(j)
                else:
                    split_a.append(i)
                    j = i
                    i += 1
            i = 1
            j = 1
            for x in words_b:
                if x.startswith("#"):
                    split_b.append(j)
                else:
                    split_b.append(i)
                    j = i
                    i += 1
        time_b = timer.time
        # output['split_mask'] = torch.as_tensor([0, *[1 if x.startswith("#") else 0
        #                                              for x in words_a],
        #                                         0, *[1 if x.startswith("#") else 0
        #                                              for x in words_b], 0])
        with timer:
            output['split_tokens'] = torch.as_tensor([0, *split_a, 0,
                                                      *(torch.as_tensor(split_b) + max(split_a)), 0])
            output['split_range'] = torch.arange(1, output['split_tokens'].max()+1)
            output["input_ids"] = torch.as_tensor([self.cls,
                                                   *sent_a['input_ids'][:limit_a], self.sep,
                                                   *sent_b['input_ids'][:limit_b], self.sep],
                                                  dtype=torch.long)
            output["token_type_ids"] = torch.cat((torch.zeros(limit_a+2, dtype=torch.long),
                                                  torch.ones(limit_b+1, dtype=torch.long)))
            output["attention_mask"] = torch.ones_like(output['input_ids'])
            output["special_tokens_mask"] = torch.as_tensor(
                self.tokenizer.get_special_tokens_mask(output['input_ids'],
                                                       already_has_special_tokens=True),
                dtype=torch.long)
            output["next_sentence_label"] = next_sent
        time_c = timer.time
        # if output['split_tokens'].any():
        #     import ipdb; ipdb.set_trace()
        return output           # , time_a, time_b, time_c

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
    _inputs = inputs.clone()
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
    _inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool() &
                      masked_indices & ~indices_replaced)
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    _inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens
    # unchanged
    return _inputs, labels


def _mask_whole_words(inputs, special_tokens_mask=None,
                      tokenizer=None, split_tokens=None,
                      split_range=None, mlm_probability=0.15, ignore_index=-1):
    inputs = inputs.clone()
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability
    # `mlm_probability`)
    pm = torch.full(split_range.shape, mlm_probability)
    pm[~split_range.bool()] = 0
    replacements = torch.bernoulli(pm)
    masked_indices = torch.empty(inputs.shape, dtype=torch.bool)
    for i in range(len(inputs)):
        repl_inds = set(torch.where(replacements[i] == 1)[0].tolist())
        masked_indices[i] = torch.as_tensor([*map(lambda x: x in repl_inds,
                                                  split_tokens[i].tolist())],
                                            dtype=bool)
    labels[~masked_indices] = ignore_index
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


def mlm_collator(batch, seq_align_len, tokenizer, pad_full=0, mask_whole_words=False):
    """Collate for MLM task

    Args:
        batch: A batch dictionary of instances. Should have 'input_ids'
        seq_align_len: Align sequences to multiple of this number
        tokenizer: tokenizer
        pad_full: pad to max_seq_len

    """
    # if isinstance(batch, tuple):
    #     batch = batch[0]
    output = {x: None for x in ["input_ids", "token_type_ids", "attention_mask",
                                "masked_lm_labels", "next_sentence_labels"]}
    lengths = [len(x['input_ids']) for x in batch]
    batch_size = len(batch)
    max_len = max(lengths)
    width = pad_full if pad_full else int(np.ceil(max_len / seq_align_len) * seq_align_len)
    size = (batch_size, width)
    input_tensor = torch.zeros(size, dtype=torch.long)
    mask_tensor = torch.zeros(size, dtype=torch.long)
    special_tokens_mask_tensor = torch.zeros(size, dtype=torch.long)
    token_type_id_tensor = torch.zeros(size, dtype=torch.long)
    if mask_whole_words:
        split_tokens_tensor = torch.zeros(size, dtype=torch.long)
        split_range_tensor = torch.zeros(size, dtype=torch.long)
    next_sentence_labels = []
    with torch.no_grad():
        for i, x in enumerate(batch):
            limit = lengths[i]
            input_tensor[i, :limit] = x['input_ids']
            mask_tensor[i, :limit] = x['attention_mask']
            special_tokens_mask_tensor[i, :limit] = x['special_tokens_mask']
            token_type_id_tensor[i, :limit] = x['token_type_ids']
            if mask_whole_words:
                split_tokens_tensor[i, :limit] = x['split_tokens']
                split_range_tensor[i, :x['split_range'].max()] = x['split_range']
            next_sentence_labels.append(x['next_sentence_label'])
        if mask_whole_words:
            output["input_ids"], output["masked_lm_labels"] = _mask_whole_words(
                input_tensor, tokenizer=tokenizer,
                special_tokens_mask=special_tokens_mask_tensor,
                split_tokens=split_tokens_tensor,
                split_range=split_range_tensor)
        else:
            output["input_ids"], output["masked_lm_labels"] = _mask_tokens(
                input_tensor, tokenizer=tokenizer,
                special_tokens_mask=special_tokens_mask_tensor)
        output["attention_mask"] = mask_tensor
        output["token_type_ids"] = token_type_id_tensor
        output["next_sentence_labels"] = torch.as_tensor(next_sentence_labels)
    return output


def get_wiki_books_loader(batch_size, num_workers, seq_align_len, shuffle=True,
                          max_seq_len=None, min_seq_len=None,
                          truncate_strategy=None):
    if max_seq_len is None:
        raise ValueError("max_seq_len cannot be None")
    mask_whole_words = True
    if mask_whole_words:
        print("Whole word masking is on")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased-whole")
    collate_fn = partial(mlm_collator, seq_align_len=seq_align_len,
                         tokenizer=tokenizer, mask_whole_words=mask_whole_words)
    data = BertDatasetBooksWiki("books-wiki-tokenized-with-tokens", tokenizer,
                                shuffle=shuffle, max_seq_len=max_seq_len,
                                min_seq_len=min_seq_len, truncate_strategy=truncate_strategy)
    loader = torch.utils.data.DataLoader(data, shuffle=False,
                                         num_workers=num_workers, drop_last=False,
                                         pin_memory=True, batch_size=batch_size,
                                         collate_fn=collate_fn)
    return loader


def get_owt_loader(num_train_examples, batch_size, num_workers, seq_align_len,
                   max_seq_len=None, min_seq_len=None, truncate_strategy=None):
    if max_seq_len is None:
        raise ValueError("max_seq_len cannot be None")
    mask_whole_words = True
    if mask_whole_words:
        print("Whole word masking is on")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased-whole")
    collate_fn = partial(mlm_collator, seq_align_len=seq_align_len,
                         tokenizer=tokenizer, mask_whole_words=mask_whole_words)
    data = BertDatasetOWT("owt-filtered-with-tokens", tokenizer, max_seq_len=max_seq_len,
                          min_seq_len=min_seq_len, truncate_strategy=truncate_strategy)
    sampler = torch.utils.data.RandomSampler(data, replacement=True,
                                             num_samples=num_train_examples)
    loader = torch.utils.data.DataLoader(data, sampler=sampler,
                                         num_workers=num_workers, drop_last=False,
                                         pin_memory=True, batch_size=batch_size,
                                         collate_fn=collate_fn)
    return loader
