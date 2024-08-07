from statistics import median


def aggregate_measurements(measurements, agg_func):
    if isinstance(measurements, dict):
        agg_measurements = {}
        for metric in measurements["0"].keys():
            values = [measurements[i][metric] for i in measurements.keys()]
            agg_measurements[metric] = agg_func(values)
        return agg_measurements
    elif isinstance(measurements, list):
        agg_measurements = {}
        for metric in measurements[0].keys():
            values = [measurement[metric] for measurement in measurements]
            agg_measurements[metric] = agg_func(values)
        return agg_measurements

if __name__ == '__main__':
    test = [{'state_dict_size': 1, 'model_to_device': 2},
            {'state_dict_size': 3, 'model_to_device': 2},
            {'state_dict_size': 2, 'model_to_device': 5},
            {'state_dict_size': 3, 'model_to_device': 2},
            {'state_dict_size': 1, 'model_to_device': 10}]
    res = aggregate_measurements(test, median)
    print(res)
