import os
import pytorch_lightning as pl
from moco import SelfSupervisedMethod
from model_params import VICRegParams

os.environ["DATA_PATH"] = "D:\\Documents\\SBB\\runs\\20220427_full_dataset\\dataset\\split.csv"


def main():
    params = VICRegParams(dataset_name='aisi')
    model = SelfSupervisedMethod(params)
    trainer = pl.Trainer(gpus=1, max_epochs=320)
    trainer.fit(model)
    trainer.save_checkpoint("vicreg.ckpt")


if __name__ == '__main__':
    main()
