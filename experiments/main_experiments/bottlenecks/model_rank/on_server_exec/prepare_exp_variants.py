import configparser

from global_utils.model_names import VISION_MODEL_CHOICES, VIT_L_32, EFF_NET_V2_L, RESNET_152, RESNET_18

if __name__ == '__main__':
    config = configparser.ConfigParser()
    sections = []

    model_search_space = VISION_MODEL_CHOICES.copy()

    # generate ini files
    for model_name in [RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32]:
        for split_level in [str(x) for x in [None, -3]]:
            # 9* 1024, because imagenette has not enough data to fill 10 batches a 1024
            for num_items in [96, 1024, 9 * 1024]:
                for dataset_type, batch_size, num_workers in [('imagenette', 128, 12),
                                                              ('imagenette_preprocessed_ssd', 256, 2)]:
                    section = f'bottleneck_analysis-model-{model_name}-items-{num_items}-split-{split_level}-dataset_type-{dataset_type}'

                    # if the split is not Note we assume by default that data is stored/cached on SSD
                    if split_level == 'None' or dataset_type == 'imagenette_preprocessed_ssd':
                        sections.append(section)

                    config[section] = {
                        'model_name': model_name,
                        'result_dir': '/repro-mount-fs/results/bottleneck-analysis',
                        'dataset_path': '/repro-mount-ssd/data/imagenette2',
                        'extract_batch_size': str(batch_size),  # chosen based on experiments
                        'classify_batch_size': '0',  # currently not used
                        'num_items': num_items,
                        'device': 'cuda',
                        'split_level': split_level,
                        'dataset_type': dataset_type,
                        'data_workers': str(num_workers),  # chosen based on experiments
                        'dummy_input_dir': '/repro-mount-ssd/data/dummy'
                    }

    with open('tmp-config.ini', 'w') as configfile:
        configfile.seek(0)
        config.write(configfile)

    with open('run-all-exp.sh', 'w') as script:
        script.seek(0)
        script.write('#!/bin/sh \nfor i in $(seq 1 5);\ndo\n')
        for section in sections:
            # use server_script_template.sh ad adjust it for your server
            cmd = f'\t./run_exp.sh {section} \n'
            script.write(cmd)
        script.write('done')
