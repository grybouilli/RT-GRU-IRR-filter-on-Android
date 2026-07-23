import onnx
import onnx_graphsurgeon as gs
import numpy as np
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("--inputs", nargs="+", type=str)
parser.add_argument("--outputs", nargs="+", type=str)
parser.add_argument("--q_to_copy", type=str, default="node_QuantizeLinear_2845")
parser.add_argument("--dq_to_copy", type=str, default="node_quantize_dequantize_496")
parser.add_argument("-o", "--output_model", type=str, default=None)

args = parser.parse_args()

if args.output_model is None:
    args.output_model = os.path.basename(args.model).split(".")[0] + "_complete.onnx"

model = onnx.load(args.model)
graph = gs.import_onnx(model)


def find_tensor(graph: gs.Graph, name: str) -> gs.Variable:
    """Find a tensor (Variable) anywhere in the graph by name."""
    for tensor in graph.tensors().values():
        if tensor.name == name:
            return tensor
    raise ValueError(f"Tensor '{name}' not found in graph")


def find_node(graph: gs.Graph, name: str) -> gs.Node:
    """Find a node by name."""
    matches = [n for n in graph.nodes if n.name == name]
    if not matches:
        raise ValueError(f"Node '{name}' not found in graph")
    return matches[0]


def copy_q_dq_nodes(
    graph: gs.Graph,
    q_source_name: str,
    dq_source_name: str,
    input_node_name: str,
    output_node_name: str,
    copy_idx: int = 0,
):
    original_qnode = find_node(graph, q_source_name)
    original_dqnode = find_node(graph, dq_source_name)

    input_node = find_node(graph, input_node_name)
    feed_tensor = input_node.outputs[0]
    print(
        f"  Feed tensor: {feed_tensor.name}, dtype={feed_tensor.dtype}, shape={feed_tensor.shape}"
    )

    # Fresh tensors for the Q→DQ chain — never reuse existing graph tensors
    orig_q_out_dtype = original_qnode.outputs[0].dtype
    q_out_tensor = gs.Variable(
        name=f"copy{copy_idx}_q_out_{q_source_name}",
        dtype=orig_q_out_dtype,
        shape=feed_tensor.shape,  # same spatial shape, just different dtype
    )
    dq_out_tensor = gs.Variable(
        name=f"copy{copy_idx}_dq_out_{dq_source_name}",
        dtype=np.float32,
        shape=feed_tensor.shape,  # Q/DQ are shape-preserving ops
    )
    # Copy Q and DQ nodes
    copied_qnode = gs.Node(
        op=original_qnode.op,
        name=f"copy{copy_idx}_{q_source_name}",
        inputs=[feed_tensor] + original_qnode.inputs[1:],
        outputs=[q_out_tensor],
        attrs=original_qnode.attrs,
    )
    copied_dqnode = gs.Node(
        op=original_dqnode.op,
        name=f"copy{copy_idx}_{dq_source_name}",
        inputs=[q_out_tensor] + original_dqnode.inputs[1:],
        outputs=[dq_out_tensor],
        attrs=original_dqnode.attrs,
    )

    graph.nodes.append(copied_qnode)
    graph.nodes.append(copied_dqnode)

    # Rewire: point whatever consumes feed_tensor downstream to dq_out_tensor instead
    graph_output_names = [out.name for out in graph.outputs]
    if output_node_name in graph_output_names:
        # Terminal output: swap the graph output entry
        for i, graph_out in enumerate(graph.outputs):
            if graph_out.name == output_node_name:
                graph.outputs[i] = dq_out_tensor
                break
    else:
        # Intermediate node: replace the input slot that was receiving feed_tensor
        output_node = find_node(graph, output_node_name)
        replaced = False
        for i, inp in enumerate(output_node.inputs):
            if inp.name == feed_tensor.name:
                output_node.inputs[i] = dq_out_tensor
                replaced = True
                break
        if not replaced:
            raise ValueError(
                f"Could not find tensor '{feed_tensor.name}' in inputs of node '{output_node_name}'. "
                f"Inputs are: {[inp.name for inp in output_node.inputs]}"
            )

    if feed_tensor.name == "output":
        feed_tensor.name = f"copy{copy_idx}_q_out_{q_source_name}_input"
    graph.outputs[0].name = "output"
    graph.cleanup().toposort()
    print(f"  Inserted Q-DQ pair between '{input_node_name}' and '{output_node_name}'")


idx = 0
for src_input, src_output in zip(args.inputs, args.outputs):
    print(f"Copying with I/O: {src_input} -> {src_output}")
    copy_q_dq_nodes(graph, args.q_to_copy, args.dq_to_copy, src_input, src_output, idx)
    idx += 1
modified_model = gs.export_onnx(graph)
onnx.checker.check_model(modified_model)  # validate before saving
onnx.save(modified_model, args.output_model)
print(f"Saved to {args.output_model}")
