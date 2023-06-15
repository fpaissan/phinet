from modules.benchmarks import benchmark

# Benchmark on GPU
benchmark(
    model="./weights/microyolo-20-epochs-alpha067/best.pt",
    imgsz=320,
    half=True,
    device="cpu",
)  # microyolo with phinet
# benchmark(model='yolov8n.pt', imgsz=320, half=False, device='cpu') # yolov8 nano
# benchmark(model="yolov8s.pt", imgsz=320, half=True, device="cpu")  # yolov8 small