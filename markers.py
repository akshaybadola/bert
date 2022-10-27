import os
import numpy as np
import datasets
import argparse
import sentence_transformers

from nltk.corpus import stopwords
from common_pyutil.functional import flatten
from common_pyutil.monitor import Timer


def get_data_and_inds(data_name):
    if data_name == "owt":
        data_path = "owt-filtered-with-tokens"
        owt = datasets.load_from_disk(data_path)
        lengths = np.load("owt_lengths.npz")["arr_0"]
        data = owt.select(np.where(lengths > 0)[0])
        inds = np.arange(len(data))
    elif data_name == "books-wiki":
        data_path = "books-wiki-tokenized-with-tokens"
        data = datasets.load_from_disk(data_path)
        inds = None
    print(f"Got data from {data_path}")
    return data, inds


def get_model(gpu):
    model_path = "sentence-transformers/all-mpnet-base-v2"
    print(f"Getting model {model_path}")
    model = sentence_transformers.SentenceTransformer(model_path)
    model = model.cuda(gpu)
    model = model.eval()
    return model


def get_owt_sentences(model, batch, gpu):
    sentences = [x.split("\n\n") for x in batch['text']]
    sentence_lengths = [len(x) for x in sentences]
    embeddings = model.encode(flatten(sentences, 1), device=f"cuda:{gpu}")
    return embeddings, sentence_lengths


def get_books_wiki_sentences(model, batch, gpu):
    sentences = batch['text']
    embeddings = model.encode(flatten(sentences, 1), device=f"cuda:{gpu}")
    return embeddings, None


def save_sentence_embeddings(data_name, batch_size, gpu):
    if data_name not in {"owt", "books-wiki"}:
        raise ValueError(f"Unknown data {data_name}")
    savedir = f"{data_name}_sentence_embeddings"
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    model = get_model(gpu)
    data, inds = get_data_and_inds(data_name)
    data_len = len(data)
    timer = Timer(True)

    j = 0
    batch = data[batch_size*j:batch_size*(j+1)]
    while (batch_size*j) < data_len:
        with timer:
            if data_name == "owt":
                embeddings, sentence_lengths = get_owt_sentences(model, batch, gpu)
            elif data_name == "books-wiki":
                embeddings, sentence_lengths = get_books_wiki_sentences(model, batch, gpu)
            if sentence_lengths:
                np.savez(f"{savedir}/{(batch_size*(j+1)):08}_sent_embedding.npz",
                         embeddings=embeddings, lengths=np.array(sentence_lengths))
            else:
                np.savez(f"{savedir}/{(batch_size*(j+1)):08}_sent_embedding.npz",
                         embeddings=embeddings)
        j += 1
        if not (j+1) % 10:
            print(f"{(j+1)} done in {timer.time}")
            timer.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Dataset name")
    parser.add_argument("--gpu", "-g", type=int)
    parser.add_argument("--batch-size", "-b", type=int)
    args = parser.parse_args()
    save_sentence_embeddings(args.data, args.batch_size, args.gpu)
