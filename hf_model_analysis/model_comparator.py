import json
from enum import Enum

import torch.nn

from global_utils.device import get_device

COMP_LEVEL = "comp-level"

DERIVED = "derived"
BASE = "base"


class LayerSimilarity(Enum):
    IDENTICAL = 1  # same name, same shape, same parameters
    SAME_SHAPE = 2  # same name and same shape
    SAME_NAME = 3  # same name but different shape
    DIFFERENT = 4  # no overlap


class LayerComparator:

    def __init__(self, base_layer, derived_layer, deep_comparison=False):
        self.base_layer = base_layer
        self.derived_layer = derived_layer
        self.deep_comparison = deep_comparison

    def _zeros_analysis_in_binary_tensor(self, tensor):
        total_zeros = 0
        total_bits = 0
        total_leading_zeros = 0
        total_trailing_zeros = 0
        total_weights = 0
        num_identical_weights = 0
        zeros_histogram = {key: 0 for key in range(33)}

        # Iterate over each element in the tensor
        for x in tensor:
            # Convert the integer to its binary representation (32-bit, padded with zeros)
            binary_rep = bin(x.item())[2:].zfill(32)

            # Count total zeros in the binary representation
            zeros_count = binary_rep.count('0')
            zeros_histogram[zeros_count] += 1

            if zeros_count == 32:
                num_identical_weights += 1

            total_zeros += zeros_count

            # Count leading and trailing zeros
            leading_zeros = len(binary_rep) - len(binary_rep.lstrip('0'))
            trailing_zeros = len(binary_rep) - len(binary_rep.rstrip('0'))

            total_leading_zeros += leading_zeros
            total_trailing_zeros += trailing_zeros

            # Add the total number of bits (32 bits for padded binary representation)
            total_bits += len(binary_rep)
            total_weights += 1

        # Calculate the percentage of zeros
        percentage = (total_zeros / total_bits) * 100 if total_bits > 0 else 0.0

        # Return results as a dictionary
        return {
            "percentage_of_zeros": percentage,
            "total_zeros": total_zeros,
            "total_leading_zeros": total_leading_zeros,
            "total_trailing_zeros": total_trailing_zeros,
            "total_bits": total_bits,
            "total_weights": total_weights,
            "identical_weights": num_identical_weights,
            "zeros_histogram": zeros_histogram
        }

    def compare_layers(self):

        self.base_name, self.base_weights = self.base_layer
        self.derived_name, self.derived_weights = self.derived_layer

        result = {
            "base-name": self.base_name,
            "derived-name": self.derived_name,
            "base-weight-shape": self.base_weights.shape,
            "derived-weight-shape": self.derived_weights.shape,

        }

        comparison_level = LayerSimilarity.DIFFERENT
        if not self.base_name == self.derived_name:
            result[COMP_LEVEL] = str(comparison_level)
            return result
        else:
            comparison_level = LayerSimilarity.SAME_NAME
            if not self.base_weights.shape == self.derived_weights.shape:
                result[COMP_LEVEL] = str(comparison_level)
                return result
            else:
                comparison_level = LayerSimilarity.SAME_SHAPE
                device = get_device("cpu")
                self.base_weights = self.base_weights.to(device)
                self.derived_weights = self.derived_weights.to(device)

                if torch.equal(self.base_weights, self.derived_weights):
                    comparison_level = LayerSimilarity.IDENTICAL
                    result[COMP_LEVEL] = str(comparison_level)
                    return result
                else:
                    result[COMP_LEVEL] = str(comparison_level)
                    if self.deep_comparison:
                        result["tensor-comp"] = self._compare_tensors(self.base_weights, self.derived_weights)
                    return result

    def _compare_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        if (tensor1.dtype == torch.long or tensor2.dtype == torch.long):
            return {}

        # Mean Squared Error (MSE)
        mse = torch.mean((tensor1 - tensor2) ** 2).item()

        # Convert the float tensors to their raw binary representation (int32)
        tensor1_int = tensor1.view(torch.int32).flatten()
        tensor2_int = tensor2.view(torch.int32).flatten()

        # XOR operation on the binary representation
        xor_result = torch.bitwise_xor(tensor1_int, tensor2_int)

        xor_zeros_percentage = self._zeros_analysis_in_binary_tensor(xor_result)

        # Differences
        diff = torch.abs(tensor1 - tensor2)
        min_diff = torch.min(diff).item()
        max_diff = torch.max(diff).item()
        avg_diff = torch.mean(diff).item()

        self.tensor_comparison_result = {
            "MSE": mse,
            "XOR Zeros": xor_zeros_percentage,
            "Min Difference": min_diff,
            "Max Difference": max_diff,
            "Average Difference": avg_diff,
        }

        print(self.tensor_comparison_result)

        return self.tensor_comparison_result

    def __repr__(self):
        return self.tensor_comparison_result


class ModelComparator:

    def __init__(self, base_model: torch.nn.Module, base_model_name: str, derived_model: torch.nn.Module,
                 derived_model_name: str):
        self.base_model = base_model
        self.derived_model = derived_model
        self.layer_comparisons = []
        self._compare_models()
        self.base_model_name = base_model_name
        self.derived_model_name = derived_model_name

    def _compare_models(self):
        # get both model's state dicts
        base_dict = self.base_model.state_dict()
        derived_dict = self.derived_model.state_dict()

        for item1, item2 in zip(base_dict.items(), derived_dict.items()):
            layer_comparator = LayerComparator(item1, item2)
            self.layer_comparisons.append(layer_comparator.compare_layers())

    def generate_comparison_report(self):
        report = f"comparing: base {self.base_model_name} with derived {self.derived_model_name}" + "\n"
        print(report)
        for layer_comp in self.layer_comparisons:
            report += json.dumps(layer_comp) + "\n"

        return report
