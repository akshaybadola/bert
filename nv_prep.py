import os
import glob
import collections
import pickle

import nltk
import numpy as np
import datasets
from common_pyutil.functional import flatten, takewhile

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

from tqdm import tqdm

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


def get_random_indx_except(max_num, ex):
    p = np.ones((max_num,))
    p[ex] = 0
    p = p / p.sum()
    return np.random.choice(np.arange(max_num), p=p)


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
    def __init__(self, all_documents, max_seq_length,
                 short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                 tokenizer):
        self.all_documents = all_documents
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

    def process_current_chunk(self, document, document_index, current_chunk, i):
        doc_len = len(document)
        a_end_indx = 1
        if len(current_chunk) >= 2:
            a_end_indx = rng.randint(1, len(current_chunk) - 1)
        tokens_a = []
        for j in range(a_end_indx):
            tokens_a.extend(current_chunk[j])

        tokens_b = []
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
            is_random_next = True
            target_b_length = self.target_seq_length - len(tokens_a)
            random_document_index = get_random_indx_except(doc_len, document_index)
            if random_document_index == document_index:
                is_random_next = False
            random_document = self.all_documents[random_document_index]
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

    def create_instances_from_document(self, document_index):
        document = self.all_documents[document_index]
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        doc_len = len(document)
        while i < doc_len:
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == doc_len - 1 or current_length >= self.target_seq_length:
                if current_chunk:
                    tokens_a, tokens_b, is_random_next, i =\
                        self.process_current_chunk(document, document_index, current_chunk, i)
                    instance = self.concat_tokens(tokens_a, tokens_b, is_random_next)
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances

    def process_all_documents(self, dupe_factor):
        self.instances = []
        for _ in range(dupe_factor):
            for document_index in range(len(self.all_documents)):
                self.instances.extend(
                    self.create_instances_from_document(document_index))
