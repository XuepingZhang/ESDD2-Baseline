import os.path

import numpy as np
import torch.utils.data as dat

from conf import data_root
from .audio import WaveReader
import csv

def make_dataloader(train=True,
                    data_kwargs=None,
                    num_workers=8,
                    batch_size=16):
    audio = {}
    ref_speech = {}
    ref_env = {}
    labels = {}

    label_map = {"spoof_spoof": [0,0,0], "bonafide_spoof": [1,0,0]
        , "spoof_bonafide": [0,1,0], "bonafide_bonafide": [1,1,0], "original":[1,1,1], "-":[-1,-1,-1]}


    with open(data_kwargs, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # 取你指定的字段
            audio[i] = os.path.join(data_root, row["audio_path"])
            labels[i] = label_map[row["label"]]

            ref_speech[i] = os.path.join(data_root,row["speech_path"])
            ref_env[i] = os.path.join(data_root, row["env_path"])

        ref_data = [ref_speech, ref_env]
        dataset = Dataset_(audio,ref_data,labels,train)


    return dat.DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=train)





class Dataset_(object):
    """
    Per Utterance Loader
    """

    def __init__(self, mix_scp="", ref_scp=None, labels=None, train=True, sample_rate=16000):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.train = train
        self.ref = [WaveReader(ref, sample_rate=sample_rate) for ref in ref_scp]
        self.labels = labels

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, index):
        key = self.mix.index_keys[index]

        mix = self.mix[key][0].astype(np.float32)
        label = np.array(self.labels[index], dtype=np.float32)
        path = self.mix[key][1]
        ref = []
        for reader in self.ref:
            r = reader[key][0].astype(np.float32)
            ref.append(r)
        return {
            "mix": mix,
            "ref": ref,
            "label": label,
            "file":path
        }
