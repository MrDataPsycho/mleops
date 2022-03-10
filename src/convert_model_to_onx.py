import torch
import hydra
import logging

from src.factory.model import ColaModel
from src.factory.data import DataModule
from src.pathconfig import PathConfig
import click

logger = logging.getLogger(__name__)
PATHFINDER = PathConfig()


@hydra.main(config_path=PATHFINDER.configs, config_name="config")
def convert_model(cfg):
    model_path = PATHFINDER.models.joinpath("best-checkpoint.ckpt")
    logger.info(f"Loading pre-trained model from: {model_path}")
    cola_model = ColaModel.load_from_checkpoint(str(model_path))
    export_path = PATHFINDER.models.joinpath("model.onnx")
    data_model = DataModule(
        cfg.processing.batch_size,
        cfg.processing.max_length
    )
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        cola_model,
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),
        export_path,  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    logger.info(
        f"Model converted successfully. ONNX format model is at: {export_path}"
    )


if __name__ == "__main__":
    convert_model()
