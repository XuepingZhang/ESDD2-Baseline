import pprint
import argparse
from trainer import Trainer_All
from libs.dataset import make_dataloader
from libs.utils import dump_json, get_logger
from model.model import Model

from conf import trainer_conf, train_data, dev_data, eval_data, checkpoint, epochs, batch_size, start_joint, test_data, \
    num_workers

logger = get_logger(__name__)


def run(args):
    gpuids = tuple(map(int, args.gpus.split(",")))

    nnet = Model()
    # nnet = ConvTasNet(**nnet_conf)
    trainer = Trainer_All(nnet,
                          gpuid=gpuids,
                          checkpoint=checkpoint,
                          resume=args.resume,
                          eval = args.eval,
                          test=args.test,
                          **trainer_conf,
                          start_joint=start_joint)

    data_conf = {
        "train": train_data,
        "dev": dev_data,
        "eval": eval_data,
        "test": test_data
    }
    for conf, fname in zip([trainer_conf, data_conf],
                           ["mdl.json", "trainer.json", "data.json"]):
        dump_json(conf, checkpoint, fname)

    train_loader = make_dataloader(train=True,
                                   data_kwargs=train_data,
                                   batch_size=batch_size,
                                   num_workers=num_workers)
    dev_loader = make_dataloader(train=False,
                                 data_kwargs=dev_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers)

    eval_loader = make_dataloader(train=False,
                                 data_kwargs=eval_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers)

    test_loader = make_dataloader(train=False,
                                 data_kwargs=test_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers)

    trainer.run(train_loader, dev_loader,eval_loader,test_loader, num_epochs=epochs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Command to start ConvTasNet training, configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus",
                        type=str,
                        default="0,1,2,3,4,5,6,7",
                        help="Training on which GPUs "
                        "(one or more, egs: 0, \"0,1\")")

    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Exist model to resume training from")
    parser.add_argument("--eval",
                        type=str,
                        default=None,
                        help="just eval best model")
    parser.add_argument("--test",
                        type=str,
                        default=None,
                        help="just eval best model")


    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    run(args)
