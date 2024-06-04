import math


def _sh_steps(num):
    steps = []
    while num > 1:
        steps.append(num)
        num = math.ceil(num / 2)
    return steps


def _baseline_boe(full_model_size, num_models, disk_speed):
    mbs = full_model_size * num_models
    return mbs / disk_speed


def _shift_boe(full_model_size, disk_speed, ssd_speed, sh_steps):
    # first sh iteration -> load form hdd
    mbs = full_model_size * sh_steps[0]
    _time = mbs / disk_speed

    # subsequent iterations -> load form SSD
    for step_size in sh_steps[1:]:
        mbs = full_model_size * step_size
        _time += mbs / ssd_speed

    return _time


def _mosix_boe(full_model_size, partial_model_size, disk_speed, ssd_speed, sh_steps):
    # first sh iteration
    # one full model
    mbs = 1 * full_model_size
    # other model parts
    mbs += (sh_steps[0] - 1) * partial_model_size
    _time = mbs / disk_speed

    # next sh iterations

    for step_size in sh_steps[1:]:
        # one full model
        mbs = 1 * full_model_size
        # other model parts
        mbs += (step_size - 1) * partial_model_size
        _time += mbs / ssd_speed

    return _time


if __name__ == '__main__':
    disk_speed = 120  # MB/s
    ssd_speed = 1100  # MB/s
    full_model_size = 240
    part_model_size = 19  # TODO lookup
    num_models = 35
    sh_steps = _sh_steps(num_models)
    print(sh_steps)

    base = _baseline_boe(full_model_size, num_models, disk_speed)
    print("basel:", base)
    print("shift:", _shift_boe(full_model_size, disk_speed, ssd_speed, sh_steps))
    mosix = _mosix_boe(full_model_size, part_model_size, disk_speed, ssd_speed, sh_steps)
    print("mosix:", mosix)

    print("factor baseline:", base / mosix)
