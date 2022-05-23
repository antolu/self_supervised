import os

import pytorch_lightning as pl

from moco import SelfSupervisedMethod
from model_params import VICRegParams

os.environ[
    "DATA_PATH"] = "D:\\Documents\\SBB\\runs\\20220427_full_dataset\\dataset\\split.csv"


def main():
    params = VICRegParams(dataset_name='aisi', batch_size=16,
                          encoder_arch='resnet101', embedding_dim=2048,
                          mlp_hidden_dim=2048, lr=0.03,
                          shuffle_batch_norm=True)
    model = SelfSupervisedMethod(params)
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    try:
        trainer.fit(model)
    except KeyboardInterrupt:
        pass
    finally:
        trainer.save_checkpoint("vicreg.ckpt")


if __name__ == '__main__':
    main()
