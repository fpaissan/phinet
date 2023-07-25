import os
import shutil

from micromind import benchmark


def bench(model, weight):

    # Benchmark on GPU
    benchmark(
        model=model,
        imgsz=320,
        device="cpu",
        half=True,
        int8=True,
    )  # microyolo with phinet

    # rename benchmark.log to weight.log
    os.rename("benchmarks.log", weight + ".log")

    # move to benchmark folder
    shutil.move(weight + ".log", "./benchmark/plots/data/half/" + weight + ".log")

    # benchmark(model='yolov8n.pt', imgsz=320, half=False, device='cpu') # yolov8 nano
    # benchmark(model="yolov8s.pt", imgsz=320, half=True, device="cpu")  # yolov8 small


if __name__ == "__main__":

    single = True

    if single:
        model = "./yolov8n.pt"
        bench(model, weight="yolov8n")
    else:
        weights = os.listdir("./benchmark/weights/_new_start/")

        for i in weights:
            print("benching: " + i + "...")
            weight = str(i)
            model = ("./benchmark/weights/_new_start/" + weight + "/weights/best.pt",)
            bench(model=model, weight=weight)
