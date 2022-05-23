import os
import pytorch_lightning as pl
from linear_classifier import LinearClassifierMethod


os.environ["DATA_PATH"] = "D:\\Documents\\SBB\\runs\\20220427_full_dataset\\dataset\\split.csv"


def main():
    linear_model = LinearClassifierMethod.from_moco_checkpoint("vicreg.ckpt",
                                                               batch_size=16,
                                                               lr=10)
    trainer = pl.Trainer(gpus=1, max_epochs=100)

    try:
        trainer.fit(linear_model)
    except KeyboardInterrupt:
        pass
    finally:
        trainer.save_checkpoint('vicreg_lin.ckpt')


if __name__ == '__main__':
    main()
