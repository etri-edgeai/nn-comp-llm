# Consistency-regularized Intermediate Layer Distillation (EACL2023 Findings)
[arXiv](https://arxiv.org/abs/2302.01530) | [EACL](https://aclanthology.org/2023.findings-eacl.12/) | [Slide](https://www.slideshare.net/JongwooKo1/slidespdf-257169659) | [Code](https://github.com/jongwooko/CR-ILD)


[**Revisiting Intermediate Layer Distillation for Compressing Language Models: An Overfitting Perspective**](https://arxiv.org/abs/2302.01530)<br/>
[Jongwoo Ko](https://sites.google.com/view/jongwooko),
Seungjoon Park,
Minchan Jeong,
Sukjin Hong, 
Euijai Ahn, 
Du-Seong Chang,
[Se-Young Yun](https://fbsqkd.github.io)<br/>

## Requirements
### Python modules
```
pip install -r requirements.txt
```

## Example to Run
### Prepare the GLUE datasets
```
python download_glue_data.py
```

### Prepare the pre-trained Language Models
For BERT experiments, you have to prepare the teacher model and student model.
You have to download the teacher and student model from huggingface.
- BERT-base : https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip


Then, you have to first fine-tune the teacher model, and then conducting ILD.
You only need pytorch_model.bin and config.json.

## Examples for script files

### Fine-tuning
```
bash run_ft_standard.sh ${task_name}
```

### CR-ILD
```
bash scripts/standard_glue_truncated_bert.sh 0 ${task_name}
```

### TinyBERT-like KD
```
bash scripts/standard_glue_truncated_bert.sh 1 ${task_name}
```

### BERT-EMD
```
bash scripts/standard_glue_truncated_bert.sh 2 ${task_name}
```

### Patient KD
```
bash scripts/standard_glue_truncated_bert.sh 3 ${task_name}
```

## References
- https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT
- https://github.com/lxk00/BERT-EMD
- https://github.com/GeondoPark/CKD
