# ESDD2 Baseline System

This repository contains the official implementation of ESDD2 challenge baseline system.

***

## ⚙️ Setup


```
git clone https://github.com/XuepingZhang/ESDD2-Baseline.git
cd ESDD2-Baseline
conda create -n ESDD2-Baseline python=3.10
conda activate ESDD2-Baseline
git clone https://github.com/facebookresearch/fairseq.git fairseq_dir
cd fairseq_dir
git checkout a54021305d6b3c
pip install --editable ./
pip install -r requirements.txt
```




***

## 🚀 Training & Evaluation

#### 🤗Download Compspoof V2 Dataset: [https://huggingface.co/datasets/XuepingZhang/ESDD2-CompSpoof-V2](https://huggingface.co/datasets/XuepingZhang/ESDD2-CompSpoof-V2)

We provide scripts for model training and evaluation:

    # Training
    python main.py

    # Evaluation on eval set
    python main.py --eval /path/best.pt

    # Test on test set
    python main.py --test /path/best.pt

You can modify the configs in conf.py to conduct more experiments.

***

## 📊 Results

| dataset  | F1-score | Precision | Recall  | Original eer | Speech eer | Env eer |
|:---------|:---------|:----------|:--------|:-------------|:-----------|:--------|
| val set  | 0.9462   | 0.9415    | 0.9521  | 0.0031       | 0.0112     | 0.0448  |
| eval set | 0.6224   | 0.6752    | 0.6579  | 0.0174       | 0.1889     | 0.2026  |
| test set | 0.6327   | 0.6851    | 0.6636  | 0.0173       | 0.1842     | 0.2006  |



#### metrics note:

- `F1-score`: the average F1-score over all five classes, providing a balanced measure of detection performance.
- `Precision`: the average Precision over all five classes, indicating the proportion of correctly predicted samples among all predicted samples.
- `Recall`: the average Recall over all five classes, indicating the proportion of correctly predicted samples among all true samples.
- `Original EER`: the Equal Error Rate computed for detecting whether an audio sample belongs to the original class.
- `Speech EER`: the Equal Error Rate computed from the speech anti-spoofing model’s predictions.
- `Env EER`: the Equal Error Rate computed from the environmental sound anti-spoofing model’s predictions.

Only the **F1-score** is used for the final ranking of submissions. All other metrics are reported **for reference purposes only** and do not affect the leaderboard ranking.


***

## ✨ Citation

If you find this work useful, please cite our paper:

```
@misc{zhang2026compspoofdatasetjointlearning,
      title={CompSpoof: A Dataset and Joint Learning Framework for Component-Level Audio Anti-spoofing Countermeasures}, 
      author={Xueping Zhang and Liwei Jin and Yechen Wang and Linxi Li and Ming Li},
      year={2025},
      eprint={2509.15804},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2509.15804}, 
}
```

## 🔏 License 

 The code is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) license.

---

## ✉️ Contact Information

For questions, issues, or collaboration inquiries, please contact:
* Email: [xueping.zhang@dukekunshan.edu.cn](xueping.zhang@dukekunshan.edu.cn)

