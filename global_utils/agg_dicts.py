import json
from statistics import mean, median


def aggregate_json_dicts(json_dicts, aggregation_func):
    # Initialize an empty dictionary to hold the aggregated values
    aggregated_dict = {}

    # Recursive function to traverse and aggregate the dictionaries
    def aggregate_helper(curr_dict, agg_dict):
        for key, value in curr_dict.items():
            # If the value is another dictionary, recursively call the function
            if isinstance(value, dict):
                # If the key doesn't exist in the aggregated dictionary, create it
                if key not in agg_dict:
                    agg_dict[key] = {}
                aggregate_helper(value, agg_dict[key])
            else:
                # If the key doesn't exist in the aggregated dictionary, create it with an empty list
                if key not in agg_dict:
                    agg_dict[key] = []
                agg_dict[key].append(value)

    # Loop through each JSON dictionary in the list
    for json_dict in json_dicts:
        # Call the recursive function to aggregate values for each dictionary
        aggregate_helper(json_dict, aggregated_dict)

    # Recursive function to compute aggregation on nested dictionaries
    def compute_aggregation(agg_dict):
        for key, value in agg_dict.items():
            if isinstance(value, dict):
                compute_aggregation(value)
            else:
                agg_dict[key] = aggregation_func(value)

    # Compute aggregation on the aggregated dictionary
    compute_aggregation(aggregated_dict)

    return aggregated_dict


if __name__ == '__main__':
    # Example usage:
    json_dicts = [
        # {'a': 1, 'b': {'a': 2, 'b': 3}, 'c': 4},
        # {'a': 1, 'b': {'a': 6, 'b': 7}, 'c': 8},
        # {'a': 1.9, 'b': {'a': 10, 'b': 11}, 'c': 12},
        {'imagenette': {96: {'None': {'end-to-end_measured': 1.0121490360470489, 'sum_of_steps': 0.9740698395958172,
                                      'sum_gpu_transfer': 0.0845524787902832, 'diff_end_minus_sum': 0.0380791964512317,
                                      'pct_diff': 3.909288112958345}}, 1024: {
            'None': {'end-to-end_measured': 2.5199783629504964, 'sum_of_steps': 2.563699871781282,
                     'sum_gpu_transfer': 0.13884435415267943, 'diff_end_minus_sum': -0.04372150883078563,
                     'pct_diff': -1.7054066785285409}}, 9216: {
            'None': {'end-to-end_measured': 15.775014910963364, 'sum_of_steps': 16.4313391905874,
                     'sum_gpu_transfer': 0.6074397439956666, 'diff_end_minus_sum': -0.6563242796240374,
                     'pct_diff': -3.9943444171611344}}}},
        {'imagenette': {96: {'None': {'end-to-end_measured': 1.0121490360470489, 'sum_of_steps': 0.9740698395958172,
                                      'sum_gpu_transfer': 0.0845524787902832, 'diff_end_minus_sum': 0.0380791964512317,
                                      'pct_diff': 3.909288112958345}}, 1024: {
            'None': {'end-to-end_measured': 2.5199783629504964, 'sum_of_steps': 2.563699871781282,
                     'sum_gpu_transfer': 0.13884435415267943, 'diff_end_minus_sum': -0.04372150883078563,
                     'pct_diff': -1.7054066785285409}}, 9216: {
            'None': {'end-to-end_measured': 15.775014910963364, 'sum_of_steps': 16.4313391905874,
                     'sum_gpu_transfer': 0.6074397439956666, 'diff_end_minus_sum': -0.6563242796240374,
                     'pct_diff': -3.9943444171611344}}}}
    ]

    # Aggregate using mean
    aggregated_mean = aggregate_json_dicts(json_dicts, 'mean')
    print("Aggregated (mean):", json.dumps(aggregated_mean, indent=4))

    # Aggregate using median
    aggregated_median = aggregate_json_dicts(json_dicts, 'median')
    print("Aggregated (median):", json.dumps(aggregated_median, indent=4))
