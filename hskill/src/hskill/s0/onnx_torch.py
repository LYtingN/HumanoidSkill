"""Execute a small ONNX graph (opset 18) with PyTorch on any device.

Covers exactly the operators of the MorphMimic v0.1 ``action_human`` subgraph. Unknown operators
raise, so a different export is rejected instead of silently mis-executed. Results are checked
against onnxruntime by ``hskill/tests/test_official_five_batch.py``.
"""
import numpy as np
import onnx
import torch
from onnx import numpy_helper

_DTYPES = {onnx.TensorProto.FLOAT: torch.float32, onnx.TensorProto.DOUBLE: torch.float64,
           onnx.TensorProto.INT64: torch.int64, onnx.TensorProto.INT32: torch.int32,
           onnx.TensorProto.BOOL: torch.bool}


def _ints(t):
    return [int(v) for v in t.tolist()]


class OnnxTorch:
    def __init__(self, model: onnx.ModelProto, device):
        if any(o.domain not in ("", "ai.onnx") or o.version > 18 for o in model.opset_import):
            raise ValueError("Only default-domain ONNX opset <= 18 is supported")
        self.device = torch.device(device)
        graph = model.graph
        self.constants = {t.name: torch.as_tensor(numpy_helper.to_array(t).copy(), device=self.device)
                          for t in graph.initializer}
        self.nodes = list(graph.node)
        self.inputs = [i.name for i in graph.input if i.name not in self.constants]
        self.outputs = [o.name for o in graph.output]
        for node in self.nodes:
            if not hasattr(self, "_" + node.op_type):
                raise NotImplementedError(f"ONNX operator {node.op_type} is not supported")
            if node.op_type == "Constant":
                self.constants[node.output[0]] = self._constant(node)

    def _constant(self, node):
        attr = {a.name: a for a in node.attribute}
        if "value" not in attr:
            raise NotImplementedError("Constant without a tensor value")
        return torch.as_tensor(numpy_helper.to_array(attr["value"].t).copy(), device=self.device)

    @torch.no_grad()
    def run(self, outputs, feeds):
        env = dict(self.constants)
        for name, value in feeds.items():
            env[name] = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value), device=self.device)
        for node in self.nodes:
            if node.op_type == "Constant":
                continue
            args = [env[n] if n else None for n in node.input]
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            result = getattr(self, "_" + node.op_type)(*args, **attrs)
            if not isinstance(result, (list, tuple)):
                result = [result]
            for name, value in zip(node.output, result):
                env[name] = value
        return [env[n] for n in (outputs or self.outputs)]

    # ------------------------------------------------------------------ operators (opset 18)

    def _Constant(self, **_):  # handled at construction
        raise AssertionError

    def _Shape(self, x, start=0, end=None):
        return torch.tensor(x.shape[start:end], dtype=torch.int64, device=self.device)

    def _Reshape(self, x, shape, allowzero=0):
        shape = _ints(shape)
        if not allowzero:
            shape = [x.shape[i] if s == 0 else s for i, s in enumerate(shape)]
        return x.reshape(shape)

    def _Unsqueeze(self, x, axes):
        axes = sorted(a % (x.dim() + len(axes)) for a in _ints(axes))
        for a in axes:
            x = x.unsqueeze(a)
        return x

    def _Expand(self, x, shape):
        shape = torch.broadcast_shapes(tuple(x.shape), tuple(_ints(shape)))
        return x.expand(shape)

    def _Concat(self, *xs, axis):
        return torch.cat(xs, dim=axis)

    def _Transpose(self, x, perm=None):
        return x.permute(perm) if perm is not None else x.permute(*reversed(range(x.dim())))

    def _Split(self, x, split=None, axis=0, num_outputs=None):
        if split is not None:
            return list(torch.split(x, _ints(split), dim=axis))
        return list(torch.tensor_split(x, num_outputs, dim=axis))

    def _Cast(self, x, to, saturate=1):
        return x.to(_DTYPES[to])

    def _Where(self, c, a, b):
        return torch.where(c, a, b)

    def _And(self, a, b):
        return a & b

    def _Add(self, a, b):
        return a + b

    def _Sub(self, a, b):
        return a - b

    def _Mul(self, a, b):
        return a * b

    def _Div(self, a, b):
        return a / b if a.is_floating_point() else torch.div(a, b, rounding_mode="trunc")

    def _Sqrt(self, x):
        return torch.sqrt(x)

    def _Tanh(self, x):
        return torch.tanh(x)

    def _Elu(self, x, alpha=1.0):
        return torch.nn.functional.elu(x, alpha)

    def _MatMul(self, a, b):
        return torch.matmul(a, b)

    def _Gemm(self, a, b, c=None, alpha=1.0, beta=1.0, transA=0, transB=0):
        a = a.transpose(-1, -2) if transA else a
        b = b.transpose(-1, -2) if transB else b
        y = alpha * (a @ b)
        return y + beta * c if c is not None else y
