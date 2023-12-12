# Calc features experiment

# Run on DES GPU Server in docker container
- `screen -S nils`
- `export PYTHONPATH=${PYTHONPATH}:/tmp/pycharm_project_635/`
- ```python
    /home/nils/.virtualenvs/model-search-paper/bin/python /tmp/pycharm_project_635/experiments/end_to_end_times/calc_features/calc_features_exp.py
    --result_dir /tmp/pycharm_project_635/experiments/end_to_end_times/calc_features/results
    --data_dir /tmp/pycharm_project_635/data/imagenette2
    --device cuda
    --num_repetitions 5
    --env_name DES-GPU-SERVER
    --use_defined_parameter_sets
```

