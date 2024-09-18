"""
This code runs the image classification training loop. It tries to support as much
as timm's functionalities as possible.

For compatibility the prefetcher, re_split and JSDLoss are disabled.

To run the training script, use this command:
    python train.py cfg/phinet.py

You can change the configuration or override the parameters as you see fit.

Authors:
    - Francesco Paissan, 2023
"""

import sys

import torch
import torch.nn as nn
from audio_datasets import prepare_esc50_loaders

import micromind as mm
from micromind.networks import PhiNet, XiNet
from micromind.utils import parse_configuration


class PhiNet(PhiNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bn0 = nn.BatchNorm2d(64)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, None]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = super().forward(x)

        return x


class AudioEncoder(nn.Module):
    def __init__(
        self,
        model,
    ) -> None:
        super().__init__()

        self.base = model


class SED(mm.MicroMind):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

        tmp = PhiNet(
            input_shape=(1, 640, 64), compatibility=True, **self.hparams.phinet_conf
        )
        tmp = AudioEncoder(tmp)
        tmp.load_state_dict(torch.load(self.hparams.name + ".ckpt"), strict=False)

        self.modules["backbone"] = tmp.base
        print("Loaded pre-trained tinyCLAP backbone.")
        self.modules["classifier"] = nn.Linear(288, 50)

        print("Number of parameters for each module:")
        print(self.compute_params())

        print("Number of MAC for each module:")
        print(self.compute_macs(hparams.input_shape))

    def preprocess(self, wavs):
        """Pre-process wavs."""
        self.hparams.spectrogram_extractor = self.hparams.spectrogram_extractor.to(
            self.device
        )
        self.hparams.logmel_extractor = self.hparams.logmel_extractor.to(self.device)
        x = self.hparams.spectrogram_extractor(wavs)
        x = self.hparams.logmel_extractor(x)

        return x

    def forward(self, batch):
        """Computes forward step for image classifier.

        Arguments
        ---------
        batch : List[torch.Tensor, torch.Tensor]
            Batch containing the images and labels.

        Returns
        -------
        Predicted class and augmented class. : Tuple[torch.Tensor, torch.Tensor]
        """
        wav, target = batch

        # pre-processing
        img = self.preprocess(wav.squeeze(1))

        emb = self.modules["backbone"](img)
        y_hat = self.modules["classifier"](emb.mean((-1, -2)))

        return (y_hat, target)

    def compute_loss(self, pred, batch):
        """Sets up the loss function and computes the criterion.

        Arguments
        ---------
        pred : Tuple[torch.Tensor, torch.Tensor]
            Predicted class and augmented class.
        batch : List[torch.Tensor, torch.Tensor]
            Same batch as input to the forward step.

        Returns
        -------
        Cost function. : torch.Tensor
        """
        criterion = nn.CrossEntropyLoss()

        return criterion(pred[0], pred[1])

    def configure_optimizers(self):
        """Configures the optimizes and, eventually the learning rate scheduler."""
        opt = torch.optim.Adam(self.modules.parameters(), lr=3e-4, weight_decay=0.0005)
        return opt


def top_k_accuracy(k=1):
    """
    Computes the top-K accuracy.

    Arguments
    ---------
    k : int
       Number of top elements to consider for accuracy.

    Returns
    -------
        accuracy : Callable
            Top-K accuracy.
    """

    def acc(pred, batch):
        if pred[1].ndim == 2:
            target = pred[1].argmax(1)
        else:
            target = pred[1]
        _, indices = torch.topk(pred[0], k, dim=1)
        correct = torch.sum(indices == target.view(-1, 1))
        accuracy = correct.item() / target.size(0)

        return torch.Tensor([accuracy]).to(pred[0].device)

    return acc


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])

    loaders = prepare_esc50_loaders(hparams)

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name
    )

    checkpointer = mm.utils.checkpointer.Checkpointer(
        exp_folder, hparams=hparams, key="loss"
    )

    mind = SED(hparams=hparams)

    top1 = mm.Metric("top1_acc", top_k_accuracy(k=1), eval_only=False)

    mind.train(
        epochs=hparams.epochs,
        datasets={"train": loaders["train"], "val": loaders["valid"]},
        metrics=[top1],
        checkpointer=checkpointer,
        debug=hparams.debug,
    )

    mind.test(datasets={"test": val_loader}, metrics=[top1])
