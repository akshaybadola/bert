import os
import glob
import json
import pickle

import numpy as np
import torch

import pyarrow as pa
import pyarrow.parquet as pq

from common_pyutil.monitor import Timer


data_keys = ['tokens', 'segment_ids', 'is_random_next', 'masked_lm_positions',
             'masked_lm_labels']


class HFData(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.keys = data_keys

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        instances = {k: [] for k in self.keys}
        ex = self.data[i]["instances"]
        for ins in ex:
            for k in instances:
                instances[k].append(ins[k])
        return instances


def collate(batch):
    keys = data_keys
    instances = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            instances[k].extend(b[k])
    return instances


def join_data(hf_data, batch_size):
    timer_1 = Timer()
    timer_2 = Timer()
    data = HFData(hf_data)
    loader = torch.utils.data.DataLoader(data, batch_size=32, num_workers=16,
                                         drop_last=False, shuffle=False,
                                         collate_fn=collate)
    keys = data_keys
    instances = {k: [] for k in keys}
    it = loader.__iter__()
    instances = {k: [] for k in keys}
    with timer_1:
        try:
            j = 0
            while True:
                with timer_2:
                    batch = it.__next__()
                print(f"Batch time for {j}: {timer_2.time}")
                with timer_2:
                    for k in keys:
                        instances[k].extend(batch[k])
                print(f"Join time for {j}: {timer_2.time}")
                j += 1
        except StopIteration:
            pass
    print("Total time: {timer_1.time}")
    return instances


def hf_to_parquet(hf_data, split_size, out_dir, batch_size, num_workers):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    data = HFData(hf_data)
    split_to_parquet(data, out_dir, split_size, batch_size, num_workers)


def split_to_parquet(data, out_dir, split_size, batch_size, num_workers):
    timer = Timer()
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers,
                                         drop_last=False, shuffle=False,
                                         collate_fn=collate)
    keys = data_keys
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with timer:
        it = iter(loader)
    print(f"Time for getting loader iterator {timer.time}")
    len_loader = len(loader)
    len_data = len(data)
    prec = 1
    while len_data := len_data // 10:
        prec += 1
    i = 0
    j = 0
    temp = {k: [] for k in keys}
    with timer:
        while True:
            try:
                batch = it.__next__()
                j += 1
            except StopIteration:
                break
            for key in keys:
                temp[key].extend(batch[key])
            if len(temp[keys[0]]) >= split_size:
                pq.write_table(pa.table({k: temp[k][:split_size] for k in keys}),
                               os.path.join(out_dir, f"data-{i:0{prec}}.pq"),
                               compression="NONE")
                for k in temp:
                    temp[k] = temp[k][split_size:]
                print(f"File number {i+1} written")
                i += 1
            if not j % 100:
                print(f"{j} out of {len_loader} batches fetched")
        pq.write_table(pa.table({k: temp[k] for k in keys}),
                       os.path.join(out_dir, f"data-{i:0{prec}}.pq"),
                       compression="NONE")
        print(f"File number {i+1} written")
    print(f"Total time taken for prep: {timer.time}")


def write_metadata(data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f"No such directory {data_dir}")
    files = [f for f in os.listdir(data_dir)
             if f.endswith("pq")]
    for f in files:
        meta_fname = os.path.join(data_dir, f.replace(".pq", ".meta"))
        if not os.path.exists(meta_fname):
            data = pq.read_table(os.path.join(data_dir, f))
            with open(meta_fname, "w") as fp:
                json.dump({"len": len(data)}, fp)


def dump_to_pq(data, out_dir, prec):
    """Dump Huggingface :code:`datasets` data to parquet files

    The datasets are generated according to Nvidia implementation which makes one data instance
    per document. That is used to generate masks while taking care that next sentence task
    doesn't use another document's sentence for that.

    Args:
        data: :code:`datasets` dataset
        out_dir: output director to place the files
        prec: number of zeros to for formatting filenames


    """
    for i in range(len(data)):
        out_file = f"{out_dir}/data-{i:0{prec}}.pq"
        if os.path.exists(out_file):
            continue
        row = {"tokens": data.data['tokens'][i].as_py(), "len": data.data['len'][i].as_py()}
        pq.write_table(pa.table(row), out_file, compression='NONE')
        print(i)


def load_from_pq(data_dir):
    """Load parquet files into memory

    Although it takes a bit of time and memory consumption is very high, the
    speedup for preprocessing is worth it.

    Args:
        data_dir: Directory where the parquet files are stored

    """
    timer = Timer(True)
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    files.sort()
    data = []
    for i, f in enumerate(files):
        with timer:
            data.append(pq.read_table(f))
        if not (i+1) % 1000:
            print(f"{i+1} out of {len(files)} done in {timer.time}s")
            timer.clear()
    return data


class ParquetData(torch.utils.data.Dataset):
    """A simple parquet files Dataset

    From each document per data instance, we extract lines and generate
    instances for MLM task. This is optimized for loading with :mod:`torch`.

    The data instances are written as batches of certain :code:`split_size`.
    See :class:`nv_prep.TrainingInstance` for how it's formatted.

    For any example we find the file index and then example index by simple
    division

    Args:
        data_dir: The dataset containing parquet files

    """

    def __init__(self, data_dir):
        self.files = glob.glob(f"{data_dir}/*")
        self.files.sort()
        self.split_size = len(pq.read_table(self.files[0]))
        self._last_data_size = len(pq.read_table(self.files[-1]))
        self._data_indx = {i: False for i in range(len(self.files))}
        self.data = {i: None for i in range(len(self.files))}
        self.keys = ['tokens', 'segment_ids', 'is_random_next', 'masked_lm_positions',
                     'masked_lm_labels']

    def _data_point(self, file_indx):
        if not self._data_indx[file_indx]:
            self.data[file_indx] = pq.read_table(self.files[file_indx])
            self._data_indx[file_indx] = True
        return self.data[file_indx]

    def __len__(self):
        return (len(self.files) - 1) * self.split_size + self._last_data_size

    def __getitem__(self, i):
        file_indx = i // self.split_size
        item_indx = i % self.split_size
        data_point = self._data_point(file_indx)
        return {k: data_point[k][item_indx].as_py() for k in self.keys}


class MultiParquetData(torch.utils.data.Dataset):
    """Muliplexed parquet files Dataset

    Like ParquetData but with multiplexed datasets.

    Return each item of dataset interleaved sequentially and proportionally.
    E.g., if there are two datasets and one outnumbers the other by 5:1 then
    5 items of first will be interleaved with 1 item of second

    Args:
        data_dirs: The dataset containing parquet files

    """

    def __init__(self, data_dirs):
        self.dsets = {}
        for data_dir in data_dirs:
            self.dsets[data_dir] = ParquetData(data_dir)
        self._lens = [len(x) for x in self.dsets.values()]
        self._data_dirs = [*self.dsets.keys()]
        self._create_indices()

    def _create_indices(self):
        lengths = self._lens
        dtype = "int64"
        inds_x = np.concatenate([np.ones(x, dtype=dtype)*i for i, x in enumerate(lengths)])
        inds_y = np.concatenate([np.arange(x) for x in lengths])
        self.inds = np.stack([inds_x, inds_y]).T

    def __len__(self):
        return sum(self._lens)

    def __getitem__(self, i):
        data_dir_indx, item_indx = self.inds[i]
        return self.dsets[self._data_dirs[data_dir_indx]][item_indx]
