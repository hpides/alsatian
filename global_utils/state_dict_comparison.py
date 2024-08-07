import torch


def _compare_state_dicts_key_wise(sd1, sd2):
    print()
    for key, t1, t2 in zip(list(sd1.keys()), list(sd1.values()), list(sd2.values())):
        if not ("running" in key or "tracked" in key):
            print(key, torch.equal(t1, t2))
    print()

