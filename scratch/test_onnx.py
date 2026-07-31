import onnx
import sys

def main():
    model = onnx.load("models/exp.onnx")
    for input in model.graph.input:
        print(input.name, end=": ")
        # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                    print(d.dim_value, end=", ")
                elif (d.HasField("dim_param")):
                    print(d.dim_param, end=", ")
                else:
                    print("?", end=", ")
        else:
            print("unknown shape", end="")
        print()

if __name__ == "__main__":
    main()
