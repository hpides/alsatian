import configparser

from global_utils.model_names import VISION_MODEL_CHOICES

if __name__ == '__main__':
    config = configparser.ConfigParser()
    sections = []

    model_search_space = VISION_MODEL_CHOICES.copy()

    # generate ini files
    for model_name in model_search_space:
        for split_level in [str(x) for x in [None, -1, -3, 25, 50, 75]]:
            # 9* 1024, because imagenette has not enough data to fill 10 batches a 1024
            for num_items in [3 * 32, 1024, 9 * 1024]:
                for dataset_type, batch_size, num_workers in [('imagenette', 128, 12),
                                                              ('imagenette_preprocessed_ssd', 256, 2)]:
                    section = f'bottleneck_analysis-model-{model_name}-items-{num_items}-split-{split_level}-dataset_type-{dataset_type}'
                    sections.append(section)

                    config[section] = {
                        'model_name': model_name,
                        'result_dir': '/mount-fs/results/bottleneck-analysis',
                        'dataset_path': '/mount-ssd/data/imagenette2',
                        'extract_batch_size': str(batch_size),  # chosen based on experiments
                        'classify_batch_size': '0',  # currently not used
                        'num_items': num_items,
                        'device': 'cuda',
                        'split_level': split_level,
                        'dataset_type': dataset_type,
                        'data_workers': str(num_workers),  # chosen based on experiments
                        'dummy_input_dir': '/mount-ssd/data/dummy'
                    }

    with open('tmp-config.ini', 'w') as configfile:
        config.write(configfile)

    with open('run-all-exp.sh', 'w') as script:
        script.write('#!/bin/sh \nfor i in $(seq 1 10);\ndo\n')
        for section in sections:
            # use server_script_template.sh ad adjust it for your server
            cmd = f'\t./run_exp.sh {section} \n'
            script.write(cmd)
        script.write('done')
