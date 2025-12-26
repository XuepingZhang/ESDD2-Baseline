import time
import numpy as np
import torch.optim as optim
import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from libs.eval_metrics import compute_speech_env_all_eer
from libs.utils import get_logger
import torch.nn as nn
import os
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score, precision_score

def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 gpuid=0,
                 optimizer_kwargs=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 resume=None,
                 eval=None,
                 no_impr=5,
                 start_joint=3):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )

        self.device = f"cuda:{gpuid[0]}"
        self.eval = eval
        self.optimizer_kwargs = optimizer_kwargs
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)


        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr
        self.start_joint = start_joint
        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    f"Could not find resume checkpoint: {resume}"
                )

            self.logger.info(f"Loading checkpoint from {resume}")
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"] + 1
            self.logger.info(f"Resume training from epoch {self.cur_epoch}")

            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)

            # ===== optimizer =====
            self.optimizer = self.make_optimizer(self.nnet)
            if "optim_state_dict" in cpt:
                self.optimizer.load_state_dict(cpt["optim_state_dict"])
                self.logger.info("Optimizer state resumed.")
            else:
                self.logger.warning("No optimizer state found in checkpoint.")

        else:
            self.nnet = nnet.to(self.device)
            self.optimizer = self.make_optimizer(self.nnet)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=factor,
            patience=patience,
            min_lr=min_lr)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))


    def make_optimizer(self, model):
        optimizer = optim.Adam([
            {"params": model.spar.parameters(), "lr": self.optimizer_kwargs['lr_spr']},
            {"params": model.aasist_all.parameters(), "lr": self.optimizer_kwargs['lr_anti']},
            {"params": model.aasist_speech.parameters(), "lr": self.optimizer_kwargs['lr_anti']},
            {"params": model.aasist_env.parameters(), "lr": self.optimizer_kwargs['lr_anti']},
        ])
        return optimizer
    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))


    def compute_loss(self, egs, jointed=False):
        raise NotImplementedError

    def only_eval(self, model_path, eval_loader):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Eval mode: model {model_path} not found")
        cpt = th.load(model_path, map_location="cpu")
        self.nnet.load_state_dict(cpt["model_state_dict"])
        self.logger.info(f"Loaded model from {model_path}, epoch {cpt['epoch']}")
        self.nnet.to(self.device)
        self.eval_fn(eval_loader, mode="eval")



    def train(self, data_loader, jointed,epoch):
        self.logger.info("Set train mode...")
        self.nnet.train()

        i = 0
        all_MSE = 0
        all_loss = 0

        for egs in tqdm(data_loader, desc=f"Train: {epoch}"):
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss, MSE = self.compute_loss(egs, jointed)
            loss.backward()

            self.optimizer.step()
            all_MSE = all_MSE + MSE
            all_loss = all_loss + loss

            i = i+1
            # if i > 4:
            #     break
        avg_loss = all_loss/i
        avg_MSE = all_MSE/i
        return avg_loss, avg_MSE



    def eval_fn(self, data_loader, mode):
        self.logger.info(f"Set {mode} mode...")
        self.nnet.eval()

        label_map = {
            (-1,-1,-1):-1,
            (0, 0, 1): 0,
            (0, 1, 1): 0,
            (1, 0, 1): 0,
            (1, 1, 1): 0,
            (1, 1, 0): 1,
            (0, 1, 0): 2,
            (1, 0, 0): 3,
            (0, 0, 0): 4,
        }

        all_labels, all_preds, file = [], [], []
        speech_scores, env_scores, all_scores, all_labels_list = [], [], [],[]


        with th.no_grad():
            for egs in tqdm(data_loader, desc=f"{mode}:"):
                egs = load_obj(egs, self.device)

                labels = egs["label"].cpu().numpy().tolist()
                true_labels = [label_map[tuple(lbl)] for lbl in labels]

                res = th.nn.parallel.data_parallel(self.nnet, egs, device_ids=self.gpuid)
                speech_, env_, res_speech_score, res_env_score, res_speech, res_env, res_all_score, h_all, h_speech_, h_env_, h_speech, h_env = res

                speech_scores.append(res_speech_score.detach().cpu().numpy())
                env_scores.append(res_env_score.detach().cpu().numpy())
                all_scores.append(res_all_score.detach().cpu().numpy())
                all_labels_list.extend(labels)

                res_speech_ = res_speech_score.argmax(dim=1).cpu().numpy()
                res_env_ = res_env_score.argmax(dim=1).cpu().numpy()
                res_all = res_all_score.argmax(dim=1).cpu().numpy()

                pred_labels = [
                    label_map[(int(s), int(e), int(a))]
                    for s, e, a in zip(res_speech_, res_env_, res_all)
                ]
                file.append(egs["file"])
                all_labels.extend(true_labels)
                all_preds.extend(pred_labels)

        if all_labels[0] == -1:
            self.logger.info("==========Evaluation Test2 Set==========")
            with open("./submission/test2_prediction.txt", "w", encoding="utf-8") as f:
                for fname, pred in zip(file, all_preds):
                    f.write(f"{fname}|{pred}\n")
            self.logger.info('./submission/test2_prediction.txt saved!')
            return

        speech_scores = np.vstack(speech_scores)
        env_scores = np.vstack(env_scores)
        all_scores = np.vstack(all_scores)

        eer_results = compute_speech_env_all_eer(
            speech_scores,
            env_scores,
            all_scores,
            all_labels
        )
        self.logger.info(f"{mode} Results - original_eer: {eer_results['EER_all']:.4f}, speech_eer: {eer_results['EER_speech']:.4f}, env_eer: {eer_results['EER_env']:.4f}")

        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average="macro")
        f1 = f1_score(all_labels, all_preds, average="macro")

        recall_per_class = recall_score(all_labels, all_preds, average=None)
        f1_per_class = f1_score(all_labels, all_preds, average=None)


        all_labels_arr = np.array(all_labels)
        all_preds_arr = np.array(all_preds)

        precision_per_class = []
        for cls in np.unique(all_labels_arr):
            pred_mask = (all_preds_arr == cls)
            if pred_mask.sum() == 0:
                precision_cls = 0.0
            else:
                precision_cls = (all_labels_arr[pred_mask] == cls).mean()
            precision_per_class.append(precision_cls)

        self.logger.info(f"{mode} Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            self.logger.info(f"Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
        return precision,recall, f1




    def run(self, train_loader, dev_loader,eval_loader,split_epoch = None, num_epochs=50):
        # check if save is OK
        if self.eval is not None:
            self.only_eval(self.eval, eval_loader)
            return
        no_impr_MSE = 0
        no_impr = 0
        best_MSE = 10000000.
        best_dev_f1 = 0.
        # make sure not inf
        self.scheduler.best = best_dev_f1
        while self.cur_epoch < num_epochs:
            self.cur_epoch += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info( "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                    cur_lr, self.cur_epoch))

            loss, MSE = self.train(train_loader,self.cur_epoch >= self.start_joint, epoch = self.cur_epoch)
            self.logger.info(f"train epoch:{self.cur_epoch}, loss: {loss}, MSE:{MSE}")

            dev_res = self.eval_fn(dev_loader,mode='dev')

            self.scheduler.step(dev_res[2])
            if dev_res[2] < best_dev_f1:
                no_impr += 1
                self.logger.info("| no impr, best = {:.4f}".format(
                    best_dev_f1))
            else:
                best_dev_f1 = dev_res[2]
                no_impr = 0
                self.save_checkpoint(best=True)

            # save last checkpoint
            self.save_checkpoint(best=False)
            if no_impr == self.no_impr:
                self.logger.info(
                    "Stop training cause no impr for {:d} epochs".format(
                        no_impr))
                break
        self.logger.info("Training for {:d}/{:d} epoches done!".format(
            self.cur_epoch, num_epochs))

        self.eval_fn(eval_loader,mode='eval')

class Trainer_All(Trainer):
    def __init__(self, *args, **kwargs):
        super(Trainer_All, self).__init__(*args, **kwargs)

        self.criterion = nn.MSELoss()

    def compute_loss(self, egs, jointed=False):
        # spks x n x S
        res = th.nn.parallel.data_parallel(self.nnet, egs, device_ids=self.gpuid)

        speech_, env_, res_speech_, res_env_, res_speech, res_env, res_all , h_all, h_speech_, h_env_, h_speech, h_env = res
        label_speech = egs['label'].T[0]
        label_env = egs['label'].T[1]
        label_all = egs['label'].T[2]


        weights = th.tensor([0.2, 0.8], device=res_all.device, dtype=th.float32)
        L_cls_all = F.cross_entropy(res_all, label_all.long(), weight=weights)

        mask = (label_all == 0)
        mask_count = mask.sum().item()

        # ===== (MSE) =====

        MSE = self.criterion(speech_,egs['ref'][0]) + self.criterion(env_,egs['ref'][1])


        # ===== class loss =====
        if jointed:
            L_cls_speech_, L_cls_env_ = 0.0, 0.0
            L_cons = 0.0
            if mask_count > 0:
                L_cls_speech_ = F.cross_entropy(res_speech_[mask], label_speech[mask].long())
                L_cls_env_ = F.cross_entropy(res_env_[mask], label_env[mask].long())


                log_p_speech_ = F.log_softmax(res_speech_[mask], dim=-1)
                p_speech = F.softmax(res_speech.detach()[mask], dim=-1)
                L_cons_speech = F.kl_div(log_p_speech_, p_speech, reduction='batchmean')

                log_p_env_ = F.log_softmax(res_env_[mask], dim=-1)
                p_env = F.softmax(res_env.detach()[mask], dim=-1)
                L_cons_env = F.kl_div(log_p_env_, p_env, reduction='batchmean')

                L_cons = L_cons_speech + L_cons_env

            return L_cls_all + 10*MSE + L_cls_speech_ + L_cls_env_ + L_cons, 10*MSE

        else:
            L_cls_speech, L_cls_env = 0.0, 0.0
            if mask_count > 0:
                L_cls_speech = F.cross_entropy(res_speech[mask], label_speech[mask].long())
                L_cls_env = F.cross_entropy(res_env[mask], label_env[mask].long())

            return L_cls_all + 10*MSE + L_cls_speech + L_cls_env, 10*MSE
