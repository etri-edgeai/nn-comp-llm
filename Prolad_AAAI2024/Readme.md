# Leveraging Normalization Layer in Adapters With Progressive Learning and Adaptive Distillation for Cross-Domain Few-Shot Learning

A replication of the paper "Leveraging Normalization Layer in Adapters With Progressive Learning and Adaptive Distillation for Cross-Domain Few-Shot Learning"

## Introduction

This repository contains the code for training a feature extractor and subsequently testing it with multiple adapters using progressive learning and adaptive distillation. The majority of the code is built upon the foundations laid by TSA [1] and Meta-Dataset [2].


## Setup

* Clone this repository.
* Set up Meta-Dataset by following the "User Instructions" in the Meta-Dataset [2] repository.
* Obtain additional data by adhering to the installation instructions in the CNAPs [3] repository.
* Initialize environment variables using the commands provided below.
```
    export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>
    export RECORDS=<the directory where tf-records of MetaDataset are stored>
```


## Train Feature Extractor
* To train feature extractor with the SDL setting, run
```
    ./scripts/train_resnet18_sdl
```
* To train feature extractor with the MDL setting( with URL [4] ), run
```
    ./scripts/train_resnet18_url.sh
```

## Test
* To fine-tuning on support set using TAD, run
```
    ./scripts/test_resnet18_prolad.sh
```

## References

[1] Li, W. H., Liu, X., & Bilen, H. (2022). Cross-domain few-shot learning with task-specific adapters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7161-7170).

[2] Triantafillou, E., Zhu, T., Dumoulin, V., Lamblin, P., Evci, U., Xu, K., ... & Larochelle, H. (2019). Meta-dataset: A dataset of datasets for learning to learn from few examples. arXiv preprint arXiv:1903.03096.

[3] Requeima, J., Gordon, J., Bronskill, J., Nowozin, S., & Turner, R. E. (2019). Fast and flexible multi-task classification using conditional neural adaptive processes. Advances in Neural Information Processing Systems, 32.

[4] Li, W. H., Liu, X., & Bilen, H. (2021). Universal representation learning from multiple domains for few-shot classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 9526-9535).



