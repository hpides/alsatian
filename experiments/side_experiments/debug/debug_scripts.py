import json


def sum_field(data, field):
    if isinstance(data, dict):
        total = 0
        for key, value in data.items():
            if key == field:
                total += value
            else:
                total += sum_field(value, field)
        return total
    elif isinstance(data, list):
        return sum(sum_field(item, field) for item in data)
    else:
        return 0


if __name__ == '__main__':
    file_trained = '/Users/nils/Downloads/des-gpu-imagenette-trained-snapshots/2024-07-19-17:14:49#des-gpu-imagenette-trained-snapshots-base-distribution-TWENTY_FIVE_PERCENT-approach-mosix-cache-CPU-snapshot-resnet18-models-36-level-STEPS_DETAILS.json'
    file_synthetic = '/Users/nils/Downloads/des-gpu-imagenette-synthetic-snapshots/2024-07-21-23:49:07#des-gpu-imagenette-synthetic-distribution-TWENTY_FIVE_PERCENT-approach-mosix-cache-CPU-snapshot-resnet18-models-35-items-2000-level-STEPS_DETAILS.json'

    for file in [file_trained, file_synthetic]:

        with open(file, 'r') as file:
            data = json.load(file)
            # data = data['measurements']["detailed_times"]["sh_rank_iteration_details_0"]
            # Calculate the sum of all "inference" fields
            total_inference = sum_field(data, "inference")

            print("end_to_end", sum_field(data, "end_to_end"),
                  "Total inference:", total_inference,
                  "load_data", sum_field(data, "load_data"),
                  "get_composed_model", sum_field(data, "get_composed_model"),
                  "data_to_device", sum_field(data, "data_to_device"),
                  "model_to_device", sum_field(data, "model_to_device")
                  )
