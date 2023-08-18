import os, random
import numpy as np
import torch

from .base import convert_examples_to_features, get_tensor_data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(args, tokenizer, logger):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.downsample:
        from .glue_1k import ColaProcessor, MnliProcessor, MnliMismatchedProcessor, MrpcProcessor, Sst2Processor
        from .glue_1k import StsbProcessor, QqpProcessor, QnliProcessor, RteProcessor, WnliProcessor
    elif args.mislabel:
        from .glue_lnl import ColaProcessor, MnliProcessor, MnliMismatchedProcessor, MrpcProcessor, Sst2Processor
        from .glue_lnl import StsbProcessor, QqpProcessor, QnliProcessor, RteProcessor, WnliProcessor
    else:
        from .glue import ColaProcessor, MnliProcessor, MnliMismatchedProcessor, MrpcProcessor, Sst2Processor
        from .glue import StsbProcessor, QqpProcessor, QnliProcessor, RteProcessor, WnliProcessor
        
    processors = {
            "cola": ColaProcessor,
            "mnli": MnliProcessor,
            "mnli-mm": MnliMismatchedProcessor,
            "mrpc": MrpcProcessor,
            "sst-2": Sst2Processor,
            "sts-b": StsbProcessor,
            "qqp": QqpProcessor,
            "qnli": QnliProcessor,
            "rte": RteProcessor,
            "wnli": WnliProcessor
        }

    output_modes = {
            "cola": "classification",
            "mnli": "classification",
            "mrpc": "classification",
            "sst-2": "classification",
            "sts-b": "regression",
            "qqp": "classification",
            "qnli": "classification",
            "rte": "classification",
            "wnli": "classification"
        }
    
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    
    processor = processors[task_name]()
    args.label_list = processor.get_labels()
    args.num_labels = len(args.label_list)
    args.output_mode = output_modes[task_name]
    
    if args.mislabel:
        train_examples = processor.get_train_examples(args.data_dir, args.mislabel_ratio)
    else:
        train_examples = processor.get_train_examples(args.data_dir)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    
    train_features = convert_examples_to_features(train_examples, args.label_list,
                                                  args.max_seq_length, tokenizer, args.output_mode, logger)
    train_data, _ = get_tensor_data(args.output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, args.label_list, args.max_seq_length, tokenizer, args.output_mode, logger)
    eval_data, eval_labels = get_tensor_data(args.output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    return train_dataloader, eval_dataloader, eval_labels