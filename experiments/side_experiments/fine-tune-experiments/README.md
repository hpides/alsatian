# Fine-tune experiment

- in this experiment we want to find out the impact on end-to-end training time under the assumption that we only train
  a certain number of layers
- scenario 
  - fixed model: ResNet50 (as proposed in surgical fine-tuning paper)
  - split points according surgical fine-tuning paper: 4 Stages + Last Layer
  - resulting training configurations
    - full tine-tuning
    - surgical -> exclusive fine-tuning
      - with caching on SSD 
      - without caching on SSD
    - "traditional" fine-tuning of last n blocks
      - with caching on SSD 
      - without caching on SSD