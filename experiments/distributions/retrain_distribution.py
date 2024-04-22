from scipy.stats import truncnorm


def normal_retrain_layer_dist_50(num_layers, num_models):
    mean = round(float(num_layers) / 2)
    std_dev = float(mean / 4)

    return _normal_retrain_layer_dist(num_layers, num_models, mean, std_dev)


def normal_retrain_layer_dist_25(num_layers, num_models):
    mean = round(float(num_layers) / 4)
    std_dev = float(mean / 2)

    return _normal_retrain_layer_dist(num_layers, num_models, mean, std_dev)


def normal_retrain_layer_dist_last_few(num_layers, num_models):
    mean = 2
    std_dev = 2

    return _normal_retrain_layer_dist(num_layers, num_models, mean, std_dev)

def _normal_retrain_layer_dist(num_layers, num_models, mean, std_dev):
    lower_bound = 0
    upper_bound = num_layers

    # Generate random numbers from a truncated normal distribution
    a = (lower_bound - mean) / std_dev
    b = (upper_bound - mean) / std_dev
    random_numbers = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=num_models, random_state=42)
    random_numbers = [round(x) for x in random_numbers]

    return random_numbers
