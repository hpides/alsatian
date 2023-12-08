# 

- two types of experiments
  1) full fine tuning (all parameters adjusted)
  2) feature extraction (only head trained)

# Run on DES GPU Server in docker container
- `screen -S nils`
- `export PYTHONPATH=${PYTHONPATH}:/tmp/pycharm_project_635/`
- ```python
    /home/nils/.virtualenvs/model-search-paper/bin/python /tmp/pycharm_project_635/experiments/end_to_end_times/fine_tuning/fine_tuning_exp.py 
    --result_dir /tmp/pycharm_project_635/experiments/end_to_end_times/fine_tuning/results
    --data_dir /tmp/pycharm_project_635/data/imagenette2
    --device cuda
    --new_num_classes 10
    --num_repetitions 5
    --env_name DES-GPU-SERVER
    --use_defined_parameter_sets
```