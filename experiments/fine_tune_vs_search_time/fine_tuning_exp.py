import argparse
import os

import torch
import torch.backends.cudnn as cudnn

from custom.data_loaders.custom_image_folder import CustomImageFolder
from custom.dataset_transfroms import imagenet_data_transforms
from custom.models.init_models import initialize_model
from experiments.fine_tune_vs_search_time.train_model import standard_training
from global_utils.model_names import MODEL_CHOICES, RESNET_18, MOBILE_V2, RESNET_50, RESNET_152, VIT_B_16, VIT_L_16
from global_utils.write_results import write_measurements_and_args_to_json_file

FULL_FINE_TUNING = 'full_fine_tuning'

FEATURE_EXTRACTION = 'feature_extraction'


def feature_extraction(args):
    model = initialize_model(args.model_name, new_num_classes=args.new_num_classes, pretrained=True,
                             freeze_feature_extractor=True)

    fine_tuning(args, model)


def full_fine_tuning(args):
    model = initialize_model(args.model_name, new_num_classes=args.new_num_classes, pretrained=True)

    fine_tuning(args, model)


def fine_tuning(args, model):
    cudnn.benchmark = True
    datasets = {
        # usa a custom data loader here to be able to adjust the number of samples
        phase: CustomImageFolder(os.path.join(args.data_dir, phase), imagenet_data_transforms[phase],
                                 number_images=num_samples)
        for phase, num_samples in zip(['train', 'val'], [args.train_size, args.val_size])
    }
    batch_size = args.batch_size
    device = torch.device(args.device)
    num_epochs = args.num_epochs
    model, measurements = standard_training(model, datasets, batch_size, device, num_epochs)
    write_measurements_and_args_to_json_file(
        measurements=measurements,
        args=args,
        dir_path=args.result_dir,
        file_id=f'model_name-{args.model_name}-fine_tuning_variant-{args.fine_tuning_variant}-train_size-{args.train_size}-val_size-{args.val_size}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parameters that need to be given
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--new_num_classes', type=int, required=True)
    # fine-tuning_variant
    parser.add_argument("--fine_tuning_variant", type=str, choices=[FULL_FINE_TUNING, FEATURE_EXTRACTION],
                        required=True)

    # parameters that can also be set by setting a flag
    parser.add_argument('--model_name', type=str, choices=MODEL_CHOICES)
    parser.add_argument('--train_size', type=int)
    parser.add_argument('--val_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)

    # flags
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_defined_parameter_sets", action="store_true")

    args = parser.parse_args()

    method_to_execute = None
    if args.fine_tuning_variant == FULL_FINE_TUNING:
        method_to_execute = full_fine_tuning
    elif args.fine_tuning_variant == FEATURE_EXTRACTION:
        method_to_execute = feature_extraction

    if args.debug:
        assert not args.use_defined_parameter_sets, "if debug is set use_defined_parameter_sets can not be set"
        args.model_name = RESNET_18
        args.train_size = 10
        args.val_size = 5
        args.batch_size = 4
        args.num_epochs = 2

        method_to_execute(args)

    elif args.use_defined_parameter_sets:
        for model_name in [MOBILE_V2, RESNET_18, RESNET_50, RESNET_152, VIT_B_16, VIT_L_16]:
            for train_size, val_size in [[3200, 800]]:
                for batch_size in [32]:
                    args.model_name = model_name
                    args.train_size = train_size
                    args.val_size = val_size
                    args.batch_size = batch_size
                    args.num_epochs = 10

                    method_to_execute(args)
    else:
        method_to_execute(args)
