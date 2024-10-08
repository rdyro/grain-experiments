import os
import functools
from pathlib import Path
import pdb
import time

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import grain.python as grain
import tensorflow as tf
import tensorflow_datasets as tfds
from sentencepiece import SentencePieceProcessor
import jax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from grain._src.python.dataset.transformations import source
from grain._src.python.dataset.transformations import prefetch
import numpy as np

def pad(x):
    return x
    #x = x[:20]
    #return np.array(x + [0] * (20 - len(x)))

if __name__ == "__main__":
    # Load the sentencepiece model
    sp = SentencePieceProcessor()

    mesh = Mesh(jax.devices("cpu"), P("x"))
    ds = tfds.builder("fashion_mnist",
        file_format=tfds.core.FileFormat.ARRAY_RECORD)
    ds.download_and_prepare()
    ds = ds.as_data_source()["train"]
    tokenizer = SentencePieceProcessor(model_proto=Path("tokenizer.v3").read_bytes())

    #itds = grain.IterDataset(ds)

    #extract_token = lambda x: tokenizer.encode(b"".join(x["tokens"]).decode("utf-8"))
    extract_image = lambda x: x["image"]

    dl = grain.DataLoader(data_source=ds,
        worker_count=4,
        worker_buffer_size=20,
        sampler=grain.SequentialSampler(len(ds), shard_options=grain.NoSharding(), seed=0),
        #sampler=grain.IndexSampler(len(ds), shard_options=grain.NoSharding()),

        #operations=[grain.MapOperation(extract_token), grain.MapOperation(pad), grain.BatchOperation(8)])
        operations=[grain.MapOperation(extract_image), grain.MapOperation(pad), grain.BatchOperation(1)]
      )
        
    t = time.time()
    for i in range(len(ds)):
        image = ds[i]["image"]
    t = time.time() - t
    print(f"Elapsed time: {t:.4e} s")

    t = time.time()
    for _ in dl:
        pass
    t = time.time() - t
    print(f"Elapsed time: {t:.4e} s")
