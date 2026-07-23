import onnx
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--output", type=str, default="output.onnx")
parser.add_argument("--entry_node", type=str, default="input")
parser.add_argument("--out_node", type=str, default="add_1_DequantizeLinear_Output")

args = parser.parse_args()

model_path = args.model
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
input_path = model_path
output_path = args.output  # I inserted the corresponding path.
input_names = [args.entry_node]
output_names = [args.out_node]
onnx.utils.extract_model(input_path, output_path, input_names, output_names)

# Print all output names
print("Model inputs:", [input.name for input in onnx_model.graph.input])
print("Model outputs:", [output.name for output in onnx_model.graph.output])

# Print all intermediate value names (all tensors in the graph)
for value_info in onnx_model.graph.value_info:
    print(value_info.name)
