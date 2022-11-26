import os
import glob
import collections
import pickle
import json
import argparse
import traceback

import nltk
import numpy as np
import torch
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import BertTokenizerFast

from common_pyutil.monitor import Timer
from common_pyutil.functional import flatten, takewhile

import parquet
from tokenization import convert_to_unicode


rng = np.random


def nv_prep_format_wiki_corpus(outfile):
    filter_sections = [["See also"], ["References"], ["External links"], ["Notes"], ["Footnotes"]]
    wiki_data = datasets.load_dataset("wikipedia", "20220301.en")
    len_wiki = len(wiki_data["train"])
    with open(outfile, "w", newline="\n") as f:
        for i, doc in enumerate(wiki_data["train"]):
            sentences = flatten(takewhile(lambda x: x not in filter_sections,
                                          map(nltk.sent_tokenize,
                                              filter(lambda x: bool(x.strip()),
                                                     doc['text'].split("\n")))))
            f.write("\n".join(sentences))
            f.write("\n\n")
            if not (i+1) % 1000:
                print(f"{i+1} out of {len_wiki} done")


def nv_prep_format_book_corpus(booksdir, outfile):
    files = glob.glob(os.path.join(booksdir, "*"))
    len_books = len(files)
    with open(outfile, "w", newline="\n") as f:
        for i, file in enumerate(files):
            with open(file, newline="\n") as infile:
                doc = infile.read()
            sentences = flatten(map(nltk.sent_tokenize,
                                    filter(lambda x: bool(x.strip()),
                                           doc.split("\n"))))
            f.write("\n".join(sentences))
            f.write("\n\n")
            if not (i+1) % 100:
                print(f"{i+1} out of {len_books} done")


def nv_prep_create_documents(input_file, outfile, tokenizer):
    all_documents = [[]]
    j = 0
    with open(input_file, "r") as reader:
        while True:
            print(j)
            line = convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()
            # Empty lines are used as document delimiters
            if not line:
                all_documents.append([])
                j += 1
            tokens = tokenizer.tokenize(line)
            if tokens:
                all_documents[-1].append(tokens)
            if not (j+1) % 100:
                print(f"{j+1} done")
    with open(outfile, "wb") as f:
        pickle.dump(all_documents, f)


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
        output_tokens[index] = masked_token
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    _keys = ["tokens", "segment_ids", "is_random_next",
             "masked_lm_positions", "masked_lm_labels"] 

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(self.tokens))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(self.masked_lm_positions))
        s += "masked_lm_labels: %s\n" % (" ".join(self.masked_lm_labels))
        s += "\n"
        return s
    
    @classmethod
    def keys(cls):
        return cls._keys

    def to_dict(self):
        s = {}
        s["tokens"] = self.tokens
        s["segment_ids"] = self.segment_ids
        s["is_random_next"] = self.is_random_next
        s["masked_lm_positions"] = self.masked_lm_positions
        s["masked_lm_labels"] = self.masked_lm_labels
        return s

    def __repr__(self):
        return self.__str__()


class NVPrep:
    def __init__(self, documents, max_seq_length,
                 short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                 tokenizer):
        self.documents = documents
        self.max_num_tokens = max_seq_length - 3
        self.target_seq_length = self.max_num_tokens
        self.short_seq_prob = short_seq_prob
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.tokenizer = tokenizer
        self.vocab_words = list(tokenizer.vocab.keys())

    def truncate_seq_pair(self, tokens_a, tokens_b):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= self.max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if rng.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def process_current_chunk(self, document, document_index, current_chunk, tokens_chunk,
                              target_seq_length, i):
        a_end_indx = 1
        if len(current_chunk) > 2:
            a_end_indx = rng.randint(1, len(current_chunk) - 1)
        tokens_a = []
        for j in range(a_end_indx):
            tokens_a.extend(current_chunk[j])

        tokens_b = []
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
            is_random_next = True
            target_b_length = target_seq_length - len(tokens_a)
            _j = 0
            random_document_index = np.random.randint(len(self.documents))
            while (random_document_index == document_index) and _j < 10:
                random_document_index = np.random.randint(len(self.documents))
            if random_document_index == document_index:
                is_random_next = False
            random_document = self.documents[random_document_index]["tokens"]
            if len(random_document) == 1:
                random_start = 0
            else:
                random_start = rng.randint(0, len(random_document) - 1)
            for j in range(random_start, len(random_document)):
                tokens_b.extend(random_document[j])
                if len(tokens_b) >= target_b_length:
                    break
            num_unused_segments = len(current_chunk) - a_end_indx
            i -= num_unused_segments  # CHECK
        else:
            is_random_next = False
            for j in range(a_end_indx, len(current_chunk)):
                tokens_b.extend(current_chunk[j])
        return tokens_a, tokens_b, is_random_next, i

    def concat_tokens(self, tokens_a, tokens_b, is_random_next):
        self.truncate_seq_pair(tokens_a, tokens_b)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        (tokens, masked_lm_positions, masked_lm_labels) =\
            create_masked_lm_predictions(tokens, self.masked_lm_prob,
                                         self.max_predictions_per_seq,
                                         self.vocab_words, rng)
        instance = TrainingInstance(tokens=tokens,
                                    segment_ids=segment_ids,
                                    is_random_next=is_random_next,
                                    masked_lm_positions=masked_lm_positions,
                                    masked_lm_labels=masked_lm_labels)
        return instance

    def process_document_subr(self, document, document_index):
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        doc_len = len(document)
        if rng.random() < self.short_seq_prob:
            target_seq_length = rng.randint(2, self.target_seq_length)
        else:
            target_seq_length = self.target_seq_length
        while i < doc_len:
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            # tokens_length += len(toks)
            if i == doc_len - 1 or current_length >= target_seq_length:
                if current_chunk:
                    tokens_a, tokens_b, is_random_next, i =\
                        self.process_current_chunk(document, document_index, current_chunk, None,
                                                   target_seq_length, i)
                    # self.process_current_chunk(document, document_index,
                    # current_chunk, tokens_chunk, i)
                    instance = self.concat_tokens(tokens_a, tokens_b, is_random_next)
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances

    def create_instances_from_document(self, document_index):
        # document = self.documents[document_index]["text"]
        document = self.documents[document_index]["tokens"]
        instances = self.process_document_subr(document, document_index)
        return instances
    
    def create_instances_from_document_books(self,example, document_index):
        document = self.documents[document_index]["text"]
        instances = self.process_document_subr(document, document_index)
        return {"instances": instances}

    def process_dataset_row(self, example, idx):
        document = example["tokens"]
        #try:
        #    instances = self.process_document_subr(document, idx)
        #except Exception as e:
        #    instances = []
        instances = self.process_document_subr(document, idx)
        return {"instances": instances}

    def process_all_documents(self, dupe_factor):
        self.instances = []
        for _ in range(dupe_factor):
            for document_index in range(len(self.documents)):
                self.instances.extend(
                    self.create_instances_from_document(document_index))


class Data(torch.utils.data.Dataset):
    def __init__(self, data, max_seq_length, short_seq_prob, masked_lm_prob,
                 max_predictions_per_seq, tokenizer):
        self.data = data
        self.keys = TrainingInstance.keys().copy()
        self.prep = NVPrep(documents=data, max_seq_length=max_seq_length,
                           short_seq_prob=short_seq_prob,
                           masked_lm_prob=masked_lm_prob,
                           max_predictions_per_seq=max_predictions_per_seq,
                           tokenizer=tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        instances = self.prep.process_dataset_row(self.data[i], i)
        if instances:
            instances = instances["instances"]
        result = {k: [] for k in self.keys}
        for x in instances:
            for k in self.keys:
                result[k].append(x.to_dict()[k])
        return result


def collate(batch):
    keys = ['tokens', 'segment_ids', 'is_random_next', 'masked_lm_positions',
            'masked_lm_labels']
    instances = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            instances[k].extend(b[k])
    return instances


def prep_books(index, max_seq_length, short_seq_prob=0.1, masked_lm_prob=0.15,
               max_predictions_per_seq=20, loader_batch_size=16, split_size=32000,
               num_workers=16):
    timer = Timer()
    tokenizer = BertTokenizerFast.from_pretrained("./bert-base-uncased-whole")
    data = datasets.load_from_disk("bookcorpus-split-filtered")
    dset = Data(data, max_seq_length, short_seq_prob,
                masked_lm_prob, max_predictions_per_seq, tokenizer)
    loader = torch.utils.data.DataLoader(dset, batch_size=loader_batch_size,
                                         num_workers=num_workers,
                                         drop_last=False, shuffle=False,
                                         collate_fn=parquet.collate)
    keys = TrainingInstance.keys().copy()
    it = iter(loader)
    out_dir = f"bookcorpus-parquet-dupe-{max_seq_length}-{index:02}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    len_loader = len(loader)
    len_data = len(data)
    prec = 1
    while len_data := len_data // 10:
        prec += 1
    i = 0
    temp = {k: [] for k in keys}
    while True:
        with timer:
            batch = it.__next__()
        print(f"Time for getting batch {timer.time}")
        with timer:
            for key in keys:
                # for item in batch[key]:
                #     temp[key].append(item)
                temp[key].extend(batch[key])
        print(f"Time for extending temp {timer.time}")
            # NOTE: This assumed batch is a list of dicts
            #       however it's only a dict because of collate function
            # for b in batch:
            #     for k in keys:
            #         temp[k].extend(b[k])
            # if len(temp[keys[0]] >= split_size):
        if len(temp[keys[0]]) >= split_size:
            pq.write_table(pa.table({k: temp[k][:split_size] for k in keys}),
                           os.path.join(out_dir, f"data-{i:0{prec}}.pq"),
                           compression="NONE")
            for k in temp:
                temp[k] = temp[k][split_size:]
            with open(os.path.join(out_dir, f"data-{i:0{prec}}.meta"), "w") as fp:
                json.dump({"len": len(data)}, fp)
            print(f"{i+1} out of {len_loader} done")
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--max-predictions-per-seq", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split-size", type=int, default=32000)
    parser.add_argument("--dupes", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()
    if args.data == "wiki":
        raise NotImplementedError
        # for i in range(args.dupes):
        #     prep_wiki(index=i, max_seq_length=args.max_seq_length)
    elif args.data == "books":
        for i in range(args.dupes):
            prep_books(index=i, max_seq_length=args.max_seq_length,
                       max_predictions_per_seq=args.max_predictions_per_seq,
                       loader_batch_size=args.batch_size,
                       split_size=args.split_size,
                       num_workers=args.num_workers)
    else:
        raise ValueError(f"Unknown data {args.data}")
