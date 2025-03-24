<h1 align="center">Alsatian: Optimizing Model Search for Deep Transfer Learning</h1>
<p align="center">This repository contains the code to our SIGMOD 2025 paper.
  For an overview of our (additional) experiments look into the experiments folder of this repo.<p/>

## General Info

- We developed our system under the name **_MOSIX_** and later renamed it to _Alsatian_.
- Large files like datasets, model snapshots, or large log files of experiments are not contained in the repo. If you
  need access please feel free to reach out to us!

## Repository Structure

- [setup](setup) contains information on the hardware and software setup we use.
- [model_search](model_search) contains the implementation of _Alsatian_(_Mosix_), the baseline, and the successive
  halving approach.
- [experiments](experiments) contains the scripts to execute and analyze/plot the experiments in our paper. We structure
  them into:
    - _main experiments_: Experiments that are described in detail in our paper.
    - _side experiments_: Experiments we ran to get additional insights but that do not directly correspond
      to an experiment in main our paper (you might find them in the appendix).
    - _micro benchmarks_: Evaluation of sub-operations that helped us making design decisions.
- [data](data) contains information on how to download, access, and prepare the datasets used in our experiments.
- [custom](custom) contains models, data loaders, and short scripts. These files are mostly slight adjustments
  or minor extensions of PyTorch files and helper scripts that made the development and analysis easier.
- [global utils](global_utils) is a collection of constants, helper methods, and scripts we use as part of Alsatian or our
  evaluation.

## Cite our work

If you use _Alsatian_ or reference our findings, please cite us.
 
```bibtex
@inproceedings{strassenburg_alsatian_2025,
  author = {Strassenburg, Nils and Glavic, Boris and Rabl, Tilmann},
  title = {Alsatian: Optimizing Model Search for Deep Transfer Learning},
  booktitle = {SIGMOD},
  year = {2025},
  volume = {3},
  number = {3 (SIGMOD)},
  article = {127},
  month = {6},
  doi = {10.1145/3725264},
  venueshort = {{SIGMOD}},
  keywords = {Sys4ML, Machine Learning, Transfer Learning, Model Search}
}
```
