import os

fs = 16000
epochs=100
batch_size=64
num_workers=8

joint=True # joint learning setting

data_root = '/work/xz464/mix_compspoof/CompSpoof/'
train_data = f'{data_root}development/metadata/train.csv'
dev_data = f'{data_root}development/metadata/val.csv'
eval_data = f'{data_root}test1/metadata/test1.csv'
test_data = f'{data_root}test2/metadata/test2.csv'


checkpoint_root=f'/work/xz464/zxp/ESDD2-Baseline/weight'
os.makedirs(checkpoint_root, exist_ok=True)

xlsr2_300m_path = f'{checkpoint_root}/xlsr2_300m.pt'

if not os.path.isfile(xlsr2_300m_path):
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"
    print(f"Downloading xlsr2_300m.pt to {xlsr2_300m_path} ...")
    urllib.request.urlretrieve(url, xlsr2_300m_path)
    print("Download finished.")


# trainer config
if joint:
    start_joint = 5 # joint learning from epoch 5
    checkpoint = f'{checkpoint_root}/epoch_{epochs}_joint_{start_joint}'
else:
    checkpoint = f'/{checkpoint_root}/epoch_{epochs}_no_joint'
    start_joint = epochs

aasist_conf = {
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
}
unet_conf = {
    'in_ch': 1,
    'unet_base_ch': 48,
    'unet_layers': 4,
    'out_ch': 2
}

adam_kwargs = {
    "lr_spr": 1e-3,
    "lr_anti": 1e-5,
    "weight_decay": 1e-5,
}
trainer_conf = {
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "no_impr": 10,
    "factor": 0.5
}