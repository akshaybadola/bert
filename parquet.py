import os
import glob
import json
import pickle

import torch

import pyarrow as pa
import pyarrow.parquet as pq

from common_pyutil.monitor import Timer


class HFData(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.keys = ['tokens', 'segment_ids', 'is_random_next', 'masked_lm_positions',
                     'masked_lm_labels']

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
    keys = ['tokens', 'segment_ids', 'is_random_next', 'masked_lm_positions',
            'masked_lm_labels']
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
    keys = ['tokens', 'segment_ids', 'is_random_next', 'masked_lm_positions',
            'masked_lm_labels']
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


def write_parquet(hf_data, batch_size, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    timer_1 = Timer()
    timer_2 = Timer()
    data = HFData(hf_data)
    loader = torch.utils.data.DataLoader(data, batch_size=32, num_workers=16,
                                         drop_last=False, shuffle=False,
                                         collate_fn=collate)
    it = loader.__iter__()
    with timer_1:
        try:
            j = 0
            while True:
                with timer_2:
                    batch = it.__next__()
                print(f"Batch time for {j}: {timer_2.time}")
                with timer_2:
                    pq.write_table(pa.table(batch), os.path.join(out_dir, f"data-{j:05}.pq"),
                                   compression="NONE")
                    with open(os.path.join(out_dir, f"data-{j:05}.meta"), "w") as fp:
                        json.dump({"len": len(data)}, fp)
                print(f"Write time for {j}: {timer_2.time}")
                j += 1
        except StopIteration:
            pass
    print("Total time: {timer_1.time}")


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

    The datasets is generated according to Nvidia implementation which makes one data instance
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
