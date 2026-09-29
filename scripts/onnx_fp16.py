"""Convert an ONNX model to fp16 weights with fp32 inputs/outputs, for ONNX Runtime Web's WebGPU backend."""
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxruntime.transformers.float16 import convert_float_to_float16


def to_fp16(src, dst):
    m = convert_float_to_float16(onnx.load(src), keep_io_types=True, disable_shape_infer=True, op_block_list=["Resize"])
    # The converter still turns Constant nodes that feed Resize scales into fp16, which is invalid ONNX.
    producers = {out: node for node in m.graph.node for out in node.output}
    for node in m.graph.node:
        if node.op_type != "Resize":
            continue
        for name in node.input[1:3]:
            const = producers.get(name) if name else None
            if const is not None and const.op_type == "Constant" and const.attribute[0].t.data_type == TensorProto.FLOAT16:
                t = const.attribute[0].t
                t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
    onnx.save(m, dst)
