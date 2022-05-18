import os
import pytorch_lightning as pl
from linear_classifier import LinearClassifierMethod

os.environ["DATA_PATH"] = "/mnt/data/Documents/SBB/runs/20220427_full_dataset/dataset/split_arch.csv"

linear_model = LinearClassifierMethod.from_moco_checkpoint("vicreg.ckpt")
trainer = pl.Trainer(gpus=1, max_epochs=100)

trainer.fit(linear_model)
