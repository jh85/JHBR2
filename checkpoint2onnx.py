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
    dynamic = "--dynamic" in sys.argv
    if dynamic:
        onnx_name = Path(pt_name).stem + "_dynamic.onnx"
    else:
        onnx_name = Path(pt_name).stem + f"_b{batch_size}.onnx"

    ckpt = torch.load(pt_name, map_location="cpu", weights_only=False)
    cfg = ShogiBT4v2Config()
    for k,v in ckpt["cfg"].items():
        if hasattr(cfg,k):
            setattr(cfg,k,v)
    model = ShogiBT4v2(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input_planes": {0: "batch"},
            "policy": {0: "batch"},
            "wdl": {0: "batch"},
            "mlh": {0: "batch"},
        }

    export_kwargs = dict(
        input_names=["input_planes"],
        output_names=["policy","wdl","mlh"],
        opset_version=18,
    )
    if dynamic:
        export_kwargs["dynamic_axes"] = dynamic_axes
        # Force legacy exporter (dynamo ignores dynamic_axes)
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, torch.randn(batch_size,48,9,9),
                      onnx_name, **export_kwargs)

    # Verify dynamic batch dim was set
    if dynamic:
        import onnx
        m = onnx.load(onnx_name)
        dim0 = m.graph.input[0].type.tensor_type.shape.dim[0]
        if dim0.dim_param:
            print(f"done (dynamic batch: {dim0.dim_param})")
        else:
            print(f"WARNING: batch dim is static ({dim0.dim_value}), not dynamic!")
            print("Fixing manually...")
            for inp in m.graph.input:
                inp.type.tensor_type.shape.dim[0].dim_param = "batch"
                inp.type.tensor_type.shape.dim[0].ClearField("dim_value")
            for out in m.graph.output:
                out.type.tensor_type.shape.dim[0].dim_param = "batch"
                out.type.tensor_type.shape.dim[0].ClearField("dim_value")
            onnx.save(m, onnx_name)
            print("Fixed — batch dim is now dynamic")
    else:
        print("done")

main()
