import os, json
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
ROOT = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset"
for name in ["yolov8n.pt"]:
    m = YOLO(os.path.join(ROOT, name))
    n = sum(p.numel() for p in m.model.parameters())/1e6
    print(json.dumps({"weights": name, "params_M": round(n,2),
                      "gflops_320": round(float(get_flops(m.model, 320)),1),
                      "gflops_640": round(float(get_flops(m.model, 640)),1)}))
