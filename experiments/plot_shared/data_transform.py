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
