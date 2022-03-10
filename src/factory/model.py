from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoModelForSequenceClassification


class ColaModel(pl.LightningModule):
    def __init__(self, model_name: str, lr=3e-5):
        """
        Create a model wrapper from pretrained model
        :param model_name: Ex: model_name="google/bert_uncased_L-2_H-128_A-2"
        """
        # lr=3e-5
        super(ColaModel, self).__init__()
        self.save_hyperparameters()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )

        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, input_ids, attention_mask, labels=None) -> Any:
        """
        Default forward operation
        :param input_ids: Default parameter for bert
        :param attention_mask: Default parameter for bert
        :param labels: Default parameter for bert
        :return: bert object with input parameter
        """
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Training abstract method
        :param batch: single batch
        :param batch_idx: batch idx
        :return: loss value as numpy float/int
        """
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx) -> Dict:
        """
        Validation abstract method
        :param batch: single bathc input
        :param batch_idx: batch idx
        :return: logit values as numpy int/float
        """
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging Metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def validation_epoch_end(self, outputs) -> None:
        """
        Run after validation
        :param outputs: output as input to store in wand db
        :return: Nothing
        """
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Defined optimizer
        :return: Nothing
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
