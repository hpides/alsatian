# Choosing parameters for batch size and number of workers

#### Experiment

- when performing inference two parameters that can significantly influence the runtime are:
    - number of workers: indicating the number of processes that are used in parallel to load and transform the input
      data
    - batch size: the number of data items that are processed at once on the GPU
- we already investigated the effects of ... in separate experiments
    - number of workers on data loading time ([data_loading](..%2Fdata_loading))
    - batch size on pure inference time ([inf_times](..%2Finf_times))
- in this experiment we want to investigate the joint effect and chose a good tuple of batch size and number of workers
  for our bottleneck experiments. We explicitly *do not* want to find the absolut best configuration for every single
  model, but rather a configuration that gives us reasonably good results for all models
- we distinguish between two settings
    - (1) the data (in our case image data) is stored as images and needs to be decoded and normalized before feeding it
      to the model which is part of the data loading process
    - (2) the data is already preprocessed and can be accessed in item granularity as binary data, this means the
      process is mainly I/O bound because there is no heavy data transformation to be done
- for every setting run an experiment measuring the end to end time and sub-times and vary the following parameters
    - data (two options see above)
    - models architecture
    - batch size
    - number of workers

#### Results

- the experiment was executed on the hardware as described in [readme.md](..%2F..%2F..%2Fexp_environment%2Freadme.md)
  and is also documented in every single result file
- the single result files can be found here: https://nextcloud.hpi.de/f/11798644
- plotting code can be found in [eval](eval), the actual plots in [plots](plots)
- we have three types of plots
    - (1) fixed batch size varying the number of workers
    - (2) fixed number of workers varying the batch size
    - (3) fixed model vary number of workers and batch size

#### Analysis

- general comment:
    - when optimizing for end to end time and choosing batch size and number of workers we have to consider
      that we should set the number of workers large enough to not be bottlenecked by data loading or at least limit
      this
      bottleneck
    - so naively one could say lets just set the number of workers to the highest possible number to be safe, but we see
      in the detailed analysis that the more workers we use the higher the overhead is to load the first batch which
      might be a significant delay when considering runs that only have a low number of batches to process in total
    - this time/delay to load the first batch is heavily influenced by the first batch size because every worker loads
      one batch independently, so doubling the batch size might double the size to load the first batch

- looking at the plots we can see trends, but no single clear winner in terms of batch_size workers combination
- this is why we take the top 3 configurations for every model, count them and sort the output
    - code in [top_n_configs.py](eval%2Ftop_n_configs.py)
- imagenette data (non preprocessed)
    - we see batch size **128** is best, and number of workers: 8, **12**, 16
    - we decided for the ones marked bold (12 also reasonably covers 16 but is not too much higher than 8)

```
{'batch_size': 128, 'num_workers': 8}: 6
{'batch_size': 128, 'num_workers': 12}: 5
{'batch_size': 128, 'num_workers': 16}: 5
{'batch_size': 256, 'num_workers': 4}: 4
{'batch_size': 128, 'num_workers': 4}: 4
{'batch_size': 256, 'num_workers': 2}: 3
{'batch_size': 256, 'num_workers': 8}: 2
{'batch_size': 32, 'num_workers': 16}: 2
{'batch_size': 32, 'num_workers': 12}: 2
{'batch_size': 128, 'num_workers': 2}: 2
{'batch_size': 256, 'num_workers': 1}: 1
```

- preprocessed data
    - we see batch size **256** is best, and number of workers: **2**, 1, 4
    - we decided for the ones marked bold which is the top ranked configuration

```
{'batch_size': 256, 'num_workers': 2}: 8
{'batch_size': 256, 'num_workers': 1}: 7
{'batch_size': 256, 'num_workers': 4}: 6
{'batch_size': 128, 'num_workers': 2}: 5
{'batch_size': 512, 'num_workers': 1}: 3
{'batch_size': 1024, 'num_workers': 1}: 2
{'batch_size': 128, 'num_workers': 1}: 2
{'batch_size': 128, 'num_workers': 4}: 1
{'batch_size': 512, 'num_workers': 2}: 1
{'batch_size': 1024, 'num_workers': 2}: 1
```

- we also analysed the accumulated average regret of every configuration and got the following results: 
- imagentte: best is 128 and 12, with an avg regret of less than 2% for the end-to-end time: 
 ```
  {'batch_size: 128 - workers: 12': 0.019583236090502287, 'batch_size: 128 - workers: 16': 0.028785222468581072, 'batch_size: 256 - workers: 12': 0.053072300037529274, 'batch_size: 256 - workers: 16': 0.057718699268782765, 'batch_size: 128 - workers: 8': 0.06420277980930635, 'batch_size: 32 - workers: 12': 0.0673992350161371, 'batch_size: 32 - workers: 16': 0.07157726943130482, 'batch_size: 256 - workers: 8': 0.09178642047804632, 'batch_size: 128 - workers: 32': 0.09207313667993018, 'batch_size: 32 - workers: 8': 0.10274798751861064, 'batch_size: 32 - workers: 32': 0.11545712411192295, 'batch_size: 256 - workers: 32': 0.12678579081206068, 'batch_size: 512 - workers: 12': 0.16153291842180265, 'batch_size: 512 - workers: 16': 0.1681921814786966, 'batch_size: 128 - workers: 48': 0.17040467503989762, 'batch_size: 32 - workers: 48': 0.17620230102495638, 'batch_size: 256 - workers: 48': 0.17791012664586106, 'batch_size: 512 - workers: 32': 0.2089495666812823, 'batch_size: 512 - workers: 8': 0.21101606376198445, 'batch_size: 256 - workers: 64': 0.21271887324229619, 'batch_size: 32 - workers: 64': 0.24640176611976952, 'batch_size: 512 - workers: 48': 0.248404949733075, 'batch_size: 128 - workers: 64': 0.2634974579031361, 'batch_size: 512 - workers: 64': 0.28187177513062783, 'batch_size: 128 - workers: 4': 0.3179604217673931, 'batch_size: 256 - workers: 4': 0.3360091039940472, 'batch_size: 32 - workers: 4': 0.3607195388048607, 'batch_size: 1024 - workers: 12': 0.3738046191762745, 'batch_size: 1024 - workers: 16': 0.3862074889087388, 'batch_size: 1024 - workers: 32': 0.41918364037767514, 'batch_size: 1024 - workers: 8': 0.4319360276372521, 'batch_size: 512 - workers: 4': 0.44164416783164134, 'batch_size: 1024 - workers: 48': 0.4554859375356735, 'batch_size: 1024 - workers: 64': 0.49246176549530923, 'batch_size: 1024 - workers: 4': 0.6812832157848802, 'batch_size: 128 - workers: 2': 1.0961198077052567, 'batch_size: 256 - workers: 2': 1.1049429189179514, 'batch_size: 32 - workers: 2': 1.1134035513538862, 'batch_size: 512 - workers: 2': 1.1359889048773568, 'batch_size: 1024 - workers: 2': 1.3701391141739598, 'batch_size: 256 - workers: 1': 2.8426774007975895, 'batch_size: 128 - workers: 1': 2.851523487857657, 'batch_size: 512 - workers: 1': 2.870361128474007, 'batch_size: 32 - workers: 1': 2.881582433039185, 'batch_size: 1024 - workers: 1': 2.940474843003122}
  ```
- imagentte prprocessed on SSD: best is 256 and 2, with an avg regret of less than 0.4% for the end-to-end time:
```
{'batch_size: 256 - workers: 2': 0.003611852200763847, 'batch_size: 256 - workers: 4': 0.009655600927440252, 'batch_size: 512 - workers: 2': 0.0210834978709067, 'batch_size: 256 - workers: 8': 0.022872897419966853, 'batch_size: 128 - workers: 2': 0.025626572913066428, 'batch_size: 512 - workers: 4': 0.025828179717039698, 'batch_size: 128 - workers: 4': 0.03132840434920609, 'batch_size: 256 - workers: 12': 0.03966621163174458, 'batch_size: 512 - workers: 8': 0.04159665805161183, 'batch_size: 128 - workers: 8': 0.04490885440483338, 'batch_size: 1024 - workers: 2': 0.057525217243325634, 'batch_size: 256 - workers: 16': 0.05920057073061133, 'batch_size: 128 - workers: 12': 0.06183245368229771, 'batch_size: 512 - workers: 12': 0.06249174902862012, 'batch_size: 1024 - workers: 4': 0.06621773557247548, 'batch_size: 128 - workers: 16': 0.0730946707655632, 'batch_size: 512 - workers: 16': 0.08223133021992415, 'batch_size: 256 - workers: 1': 0.08303399185860873, 'batch_size: 1024 - workers: 8': 0.0839594194334282, 'batch_size: 512 - workers: 1': 0.09460508682782144, 'batch_size: 1024 - workers: 12': 0.09957252805198481, 'batch_size: 1024 - workers: 16': 0.10781211627564603, 'batch_size: 128 - workers: 1': 0.11061507715560426, 'batch_size: 32 - workers: 2': 0.1193409914962137, 'batch_size: 1024 - workers: 1': 0.12397983405422476, 'batch_size: 32 - workers: 4': 0.12448406312499061, 'batch_size: 256 - workers: 32': 0.13467607922631816, 'batch_size: 512 - workers: 32': 0.13709591183459482, 'batch_size: 32 - workers: 8': 0.13721187211475838, 'batch_size: 128 - workers: 32': 0.14439880771923166, 'batch_size: 32 - workers: 12': 0.1476889923884294, 'batch_size: 1024 - workers: 32': 0.15861675079442888, 'batch_size: 32 - workers: 16': 0.15891395006858552, 'batch_size: 512 - workers: 48': 0.18750208682696745, 'batch_size: 256 - workers: 48': 0.19092326674557192, 'batch_size: 32 - workers: 1': 0.20411886390972814, 'batch_size: 1024 - workers: 48': 0.20650224331602138, 'batch_size: 128 - workers: 48': 0.20954250677026864, 'batch_size: 32 - workers: 32': 0.21504593843322242, 'batch_size: 512 - workers: 64': 0.2336550855522965, 'batch_size: 256 - workers: 64': 0.23749937942193564, 'batch_size: 1024 - workers: 64': 0.2525897417896963, 'batch_size: 32 - workers: 48': 0.26839907732413987, 'batch_size: 128 - workers: 64': 0.2728789490415629, 'batch_size: 32 - workers: 64': 0.33173713681374906}
```




