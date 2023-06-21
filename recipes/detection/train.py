from micromind import PhiNet

from yolo.model import microYOLO
from yolo.microyolohead import Microhead


def train_nn():

    # define backbone
    backbone = PhiNet(
        input_shape=(3, 320, 320),
        alpha=0.67,
        num_layers=6,
        beta=1,
        t_zero=4,
        include_top=False,
        num_classes=80,
        compatibility=False,
    )
    # define head
    head = Microhead()

    # load a model
    model = microYOLO(
        backbone=backbone, head=head, task="detect", nc=80
    )  # build a new model from scratch DEFAULT_CFG

    # Train the model
    model.train(data="coco128.yaml", epochs=1, imgsz=320, device="cpu", task="detect")
    model.export()


if __name__ == "__main__":
    train_nn()