def compute_eers(
    speech_scores,
    env_scores,
    original_scores,
    labels
):
    """
    speech_scores: [N, 2] tensor / ndarray
    env_scores:    [N, 2]
    all_scores:    [N, 2]
    labels:        list of [speech, env, all]
    """



    labels = np.asarray(labels)
    speech_labels = labels[:, 0]
    env_labels = labels[:, 1]
    original_labels = labels[:, 2]

    # ---------- all EER ----------
    eer_original = compute_eer(original_scores, original_labels)
    speech_scores_f = speech_scores
    speech_labels_f = speech_labels

    env_scores_f = env_scores
    env_labels_f = env_labels

    eer_speech = compute_eer(speech_scores_f, speech_labels_f)
    eer_env = compute_eer(env_scores_f, env_labels_f)

    return {
        "EER_speech": eer_speech,
        "EER_env": eer_env,
        "EER_original": eer_original,
    }

import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(scores, labels):
    """
    scores: list or np.ndarray, shape [N], (score for class 1)
    labels: list or np.ndarray, shape [N], 0/1
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return eer
