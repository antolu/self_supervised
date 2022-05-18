import os
import pytorch_lightning as pl
from moco import SelfSupervisedMethod
from model_params import VICRegParams

os.environ["DATA_PATH"] = "/mnt/data/Documents/SBB/runs/20220427_full_dataset/dataset/split_arch.csv"


def main():
    params = VICRegParams(
        dataset_name="aisi",
        encoder_arch='resnet101',
        shuffle_batch_norm=True,
        gather_keys_for_queue=True,
        # transform_apply_blur=False,
        mlp_hidden_dim=2048,
        dim=2048,
        embedding_dim=2048,
        batch_size=16,
        lr=0.01,
        final_lr_schedule_value=0,
        weight_decay=1e-4,
        lars_warmup_epochs=10,
        lars_eta=0.02
    )
    model = SelfSupervisedMethod(params)
    trainer = pl.Trainer(gpus=1, max_epochs=50)
    trainer.fit(model)
    trainer.save_checkpoint("vicreg.ckpt")


if __name__ == '__main__':
    main()
