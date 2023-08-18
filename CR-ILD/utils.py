import os

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import math
import json
import random
import numpy as np
from collections import namedtuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import pickle
import collections
from torch.nn import CrossEntropyLoss, KLDivLoss, MSELoss
from losses import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import logging
#from apex.parallel import DistributedDataParallel as DDP

def pkd_initialization(teacher, student):
    assert len(teacher.bert.encoder.layer) > len(student.bert.encoder.layer)
    student_dict = student.state_dict()
    pretrained_dict = {}
    for k, v in teacher.state_dict().items():
        if 'qa_outputs' in k:
            continue
        if k in student_dict:
            pretrained_dict[k] = v
    student_dict.update(pretrained_dict)
    student.load_state_dict(student_dict)
    return student
    
def classifier_reuse_initialization(teacher, student):
    student_dict = student.state_dict()
    pretrained_dict = {}
    
    for k, v in student.state_dict().items():
        if 'qa_outputs' in k:
            continue
        if k in student_dict:
            pretrained_dict[k] = v
    
    for k, v in teacher.state_dict().items():
        if 'qa_outputs' in k:
            continue
        if k in student_dict:
            if "classifier" in k:
                pretrained_dict[k] = v
            elif "bert.pooler" in k:
                pretrained_dict[k] = v
    
    student_dict.update(pretrained_dict)
    student.load_state_dict(student_dict)
    
    for name, param in student.named_parameters():
        if "classifier" in name or "bert.pooler" in name:
            param.requires_grad = False
    
    return student
    
