import math
from typing import Dict, Optional
import logging

import attr
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import AttributeDict
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from . import utils


log = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class LinearClassifierMethodParams:
    # encoder model selection
    encoder_arch: str = "resnet18"
    embedding_dim: int = 512

    # data-related parameters
    dataset_name: str = "stl10"
    batch_size: int = 256

    # optimization parameters
    lr: float = 30.0
    momentum: float = 0.9
    weight_decay: float = 0.0
    max_epochs: int = 100

    # data loader parameters
    num_data_workers: int = 4
    drop_last_batch: bool = True
    pin_data_memory: bool = True
    multi_gpu_training: bool = False

    pretrained: bool = False

    class_weight: Dict[int, float] = {}


class LinearClassifierMethod(pl.LightningModule):
    model: torch.nn.Module
    dataset: utils.DatasetBase
    hparams: AttributeDict

    def __init__(
            self,
            hparams: LinearClassifierMethodParams = None,
            **kwargs,
    ):
        super().__init__()

        if hparams is None:
            hparams = self.params(**kwargs)
        elif isinstance(hparams, dict):
            hparams = self.params(**hparams, **kwargs)

        self.hparams.update(attr.asdict(hparams))

        # actually do a load that is a little more flexible
        self.model = utils.get_encoder(hparams.encoder_arch,
                                       hparams.dataset_name,
                                       hparams.pretrained)

        self.dataset = utils.get_class_dataset(hparams.dataset_name)

        # self.classifier = torch.nn.Linear(hparams.embedding_dim, self.dataset.num_classes)
        self.classifier = utils.MLP(hparams.embedding_dim,
                                    self.dataset.num_classes, 2048, 2,
                                    dropout=0.8)

        self.temperature = torch.nn.Parameter(torch.ones(1))

        class_weights = [i if i not in self.hparams.class_weights
                         else self.hparams.class_weights[i]
                         for i in range(self.dataset.num_classes)]
        self.class_weights = torch.Tensor(class_weights)

    def load_model_from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if not k.startswith("model."):
                del state_dict[k]
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            embedding = self.model(x)

        logits = self.classifier(embedding)

        with torch.no_grad():
            temp_scaled_logits = self.temperature_scale(logits)

        return temp_scaled_logits

    def temperature_scale(self, logits: torch.Tensor):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0),
                                                           logits.size(1))
        return logits / temperature

    def training_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        acc1 = utils.calculate_accuracy(y_hat, y, topk=(1,))

        log_data = {"step_train_loss": loss, "step_train_acc1": acc1[0]}
        self.log_dict(log_data)
        return {"loss": loss, "log": log_data}

    def validation_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        y_hat = self.forward(x)
        acc1 = utils.calculate_accuracy(y_hat, y, topk=(1,))
        return {
            "valid_loss": F.cross_entropy(y_hat, y, weight=self.class_weights),
            "valid_acc1": acc1[0],
            'predictions': torch.max(y_hat, dim=1)[1],
            'labels': y,
            # "valid_acc5": acc5,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        avg_acc1 = torch.stack([x["valid_acc1"] for x in outputs]).mean()
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        predictions = torch.cat(
            [x['predictions'] for x in outputs]).cpu().numpy()
        # avg_acc5 = torch.stack([x["valid_acc5"] for x in outputs]).mean()

        report = classification_report(labels, predictions, output_dict=True,
                                       zero_division=0)
        report_data = {}
        for cls, data in {k: v
                          for k, v in report.items()
                          if k not in ('accuracy', 'macro avg', 'weighted_avg')
                          }.items():
            report_data[f'{cls}_precision'] = data['precision']
            report_data[f'{cls}_recall'] = data['recall']

        log_data = {"valid_loss": avg_loss, "valid_acc1": avg_acc1}
        self.log_dict(log_data, prog_bar=True)
        self.log_dict(report_data)
        return {
            "val_loss": avg_loss,
            "log": log_data,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        milestones = [math.floor(self.hparams.max_epochs * 0.6),
                      math.floor(self.hparams.max_epochs * 0.8)]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                 milestones)
        return [optimizer], [self.lr_scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.dataset.get_train(),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.get_validation(),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
        )

    @classmethod
    def params(cls, **kwargs) -> LinearClassifierMethodParams:
        return LinearClassifierMethodParams(**kwargs)

    @classmethod
    def from_moco_checkpoint(cls, checkpoint_path,
                             use_moco_hparams: bool = False, **kwargs):
        """ Loads hyperparameters and model from moco checkpoint """
        checkpoint = torch.load(checkpoint_path)
        moco_hparams = checkpoint["hyper_parameters"]

        if not use_moco_hparams:
            moco_hparams = {k: moco_hparams[k] for k in ('encoder_arch',
                                                         'embedding_dim',
                                                         'dataset_name')}
        params = cls.params(
            **moco_hparams,
            **kwargs,
        )
        model = cls(params)
        model.load_model_from_checkpoint(checkpoint_path)
        return model

    @classmethod
    def from_trained_checkpoint(cls, checkpoint_path: str, **kwargs):
        checkpoint = torch.load(checkpoint_path)
        moco_hparams = checkpoint["hyper_parameters"]

        moco_hparams = {k: moco_hparams[k] for k in ('encoder_arch',
                                                     'embedding_dim',
                                                     'dataset_name')}
        params = cls.params(
            **moco_hparams,
            **kwargs,
        )
        model = cls(params)

        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if not (k.startswith('model.') or k.startswith('classifier.')):
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        return model

    def set_temperature(self, valid_loader: Optional[DataLoader] = None):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        if valid_loader is None:
            valid_loader = self.val_dataloader()

        nll_criterion = torch.nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input_, label in valid_loader:
                input_, label = input_.to(self.device), label.to(self.device)

                logits = self.model(input_)

                logits_list.append(logits)
                labels_list.append(label)

            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()

        log.info('Before temperature - NLL: {:.3f}, ECE: {:.3f}'.format(
            before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits),
                                              labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits),
                                              labels).item()
        log.info('Optimal temperature: {:.3f}'.format(self.temperature.item()))
        log.info('After temperature - NLL: {:.3f}, ECE: {:.3f}'.format(
            after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins: int = 15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) \
                     * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(
                    avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
