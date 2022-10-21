import sys
import numpy as np
import datasets
import sentence_transformers

from nltk.corpus import stopwords
from common_pyutil.functional import flatten


def get_data_and_inds():
    owt = datasets.load_from_disk("owt-filtered-with-tokens")
    lengths = np.load("owt_lengths.npz")["arr_0"]
    data = owt.select(np.where(lengths > 0)[0])
    inds = np.arange(len(data))
    return data, inds


def get_model(gpu):
    model = sentence_transformers.SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model = model.cuda(gpu)
    model = model.eval()
    return model


def save_sentence_embeddings(gpu):
    model = get_model(gpu)
    data, inds = get_data_and_inds()
    batch_size = 512
    data_len = len(data)

    j = 0
    batch = data[batch_size*j:batch_size*(j+1)]
    while (batch_size*j) < data_len:
        sentences = [x.split("\n\n") for x in batch['text']]
        sentence_lengths = [len(x) for x in sentences]
        embeddings = model.encode(flatten(sentences, 1), device=f"cuda:{gpu}")
        np.savez(f"./owt_sentence_embeddings/{(batch_size*(j+1)):08}_sent_embedding.npz",
                 embeddings=embeddings, lengths=np.array(sentence_lengths))
        j += 1
        if not (j+1) % 100:
            print(f"{(j+1)} done")


if __name__ == '__main__':
    save_sentence_embeddings(int(sys.argv[1]))
