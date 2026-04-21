import sys
sys.path.insert(0, ".")
from pathlib import Path
import torch
from shogi_model_v2 import ShogiBT4v2, ShogiBT4v2Config

def main():
    if len(sys.argv) < 3:
        print(f"{sys.argv[0]} checkpoint_name batch_size")
        return
    pt_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    onnx_name = Path(pt_name).stem + f"_b{batch_size}.onnx"

    ckpt = torch.load(pt_name, map_location="cpu", weights_only=False)
    cfg = ShogiBT4v2Config()
    for k,v in ckpt["cfg"].items():
        if hasattr(cfg,k):
            setattr(cfg,k,v)
    model = ShogiBT4v2(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    torch.onnx.export(model, torch.randn(batch_size,48,9,9),
                      onnx_name,
                      input_names=["input_planes"],
                      output_names=["policy","wdl","mlh"],
                      opset_version=18)
    print("done")

main()
