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

def mlm_trains(args, student_model, input_ids, input_masks, segment_ids, lm_label_ids, is_next):
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    mlm_loss, nsp_loss = 0.0, 0.0
    
    # do not use distillation
    prediction_scores, seq_relationship_score, _, _, _ = student_model(
        input_ids, segment_ids, input_masks, is_student=False)
    mlm_loss = loss_fct(
        prediction_scores.view(-1, len(args.vocab_list)), lm_label_ids.view(-1))
    nsp_loss = loss_fct(
        seq_relationship_score.view(-1, 2), is_next.view(-1))
    loss = mlm_loss + nsp_loss
    
    if args.gradient_accumulation_steps > 1:
        mlm_loss = mlm_loss / args.gradient_accumulation_steps
        nsp_loss = nsp_loss / args.gradient_accumulation_steps
        loss = loss / args.gradient_accumulation_steps
        
    return loss, mlm_loss, nsp_loss

def kd_trains(args, teacher_model, student_model, input_ids, input_masks, segment_ids,
                 label_ids, output_mode="classification"):
    loss_fct = CrossEntropyLoss() if output_mode=="classification" else MSELoss()
    if args.mode in ["KD", "MixKD"]:
        kd_fct = DistillKL(T=args.temperature)
    elif args.mode=="MixCons":
        kd_fct = ConsistencyKL(T=args.temperature)
    elif args.mode=="PKD":
        kd_fct = PKD_Loss(p=2, normalize=False)
    else:
        raise NotImplementedError
    
    if args.use_mixup:
        lam = np.random.beta(1.0, 1.0)
        lam = max(lam, 1-lam)
    else:
        lam = None
    
    # distillation
        student_outputs = \
            student_model(input_ids, segment_ids, input_masks, mixup=args.use_mixup, lam=lam, is_student=True)
        student_logits = student_outputs[0]
        student_reps = student_outputs[4]
        
    with torch.no_grad():
        teacher_outputs = \
            teacher_model(input_ids, segment_ids, input_masks, mixup=args.use_mixup, lam=lam, is_student=False)
        teacher_logits = teacher_outputs[0]
        teacher_reps = teacher_outputs[4]
        
    if args.mode in ["KD", "MixKD"]:
        kd_loss = kd_fct(student_logits, teacher_logits, mode=output_mode)
    elif args.mode == "PKD":
        new_teacher_reps = teacher_reps[-1]
        new_student_reps = student_reps[-1]
        new_teacher_reps = new_teacher_reps.unsqueeze(1)[:,:,0,:]
        new_student_reps = new_student_reps.unsqueeze(1)[:,:,0,:]
        kd_loss = kd_fct(new_student_reps, new_teacher_reps)
        args.alpha = 1.0
        
    elif args.mode == "MixCons":
        kd_loss = kd_fct(student_logits, teacher_logits, teacher_logits[index], 
                         lam, mode=output_mode)
        
    cls_loss = 0.
    loss = kd_loss
    return loss, cls_loss, kd_loss
        

def pkd_trains(args, teacher_model, student_model, input_ids, input_masks, segment_ids,
               label_ids, output_mode="classification"):
    loss_fct = CrossEntropyLoss() if output_mode=="classification" else MSELoss()
    kd_fct = DistillKL(T=args.temperature)
    pkd_fct = PKD_Loss(p=2, normalize=True)
    
    # distillation kd and pkd
    student_outputs = student_model(input_ids, segment_ids, input_masks, is_student=True)
    student_logits, student_qs, student_ks, student_reps = \
            student_outputs[0], student_outputs[1], student_outputs[2], student_outputs[4]
    
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids, segment_ids, input_masks)
        teacher_logits, teacher_qs, teacher_ks, teacher_reps = \
                teacher_outputs[0], teacher_outputs[1], teacher_outputs[2], teacher_outputs[4]
        if args.mode=="PKD":
            teacher_reps = [teacher_rep for teacher_rep in teacher_reps]
        elif args.mode=="LTI_P":
            teacher_reps = [teacher_reps[-1] for _ in teacher_reps]
    
    teacher_layer_num = len(teacher_reps) - 1
    student_layer_num = len(student_reps) - 1
    assert teacher_layer_num % student_layer_num == 0
    layers_per_block = int(teacher_layer_num / student_layer_num)
    
    if args.l2l:
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)][1:]
        new_student_reps = [student_rep for student_rep in student_reps][1:]
        new_teacher_reps = torch.stack(new_teacher_reps, dim=1)[:,:,0,:]
        new_student_reps = torch.stack(new_student_reps, dim=1)[:,:,0,:]
    else:
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)][-1]
        new_student_reps = [student_rep for student_rep in student_reps][-1]
        new_teacher_reps = new_teacher_reps.unsqueeze(1)[:,:,0,:]
        new_student_reps = new_student_reps.unsqueeze(1)[:,:,0,:]
    
    cls_loss = loss_fct(student_logits, label_ids)
    kd_loss = kd_fct(student_logits, teacher_logits, mode=output_mode)
    pkd_loss = pkd_fct(new_teacher_reps, new_student_reps)
    
    if args.logit_distill:
        loss = (1 - args.alpha) * cls_loss + args.alpha * kd_loss + args.beta * pkd_loss
    else:
        loss = pkd_loss
    return loss, cls_loss, kd_loss, pkd_loss
            
def emdbert_trains(args, teacher_model, student_model, input_ids, input_masks, segment_ids,
                   label_ids, emd_fct, output_mode="classification"):
    """
    https://github.com/lxk00/BERT-EMD/blob/5f9f52d95ec63bec9f25dab6828f2c958bad16e8/bert-emd/emd_task_distill.py#L432
    """
    def compute_relations(objs1, objs2, head_size):
        rels = []
        for obj1, obj2 in zip(objs1, objs2):
            rel = torch.matmul(obj1, obj2.transpose(-1, -2))
            rel = rel / math.sqrt(head_size)
            rels.append(rel)
        return rels
    
    kd_fct = DistillKL(T=args.temperature)
    
    rep_loss, att_loss, emb_loss = 0., 0., 0.
    student_outputs = student_model(input_ids, segment_ids, input_masks, is_student=True)
    student_logits, student_qs, student_ks, student_reps = \
            student_outputs[0], student_outputs[1], student_outputs[2], student_outputs[4]
    
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids, segment_ids, input_masks)
        teacher_logits, teacher_qs, teacher_ks, teacher_reps = \
                teacher_outputs[0], teacher_outputs[1], teacher_outputs[2], teacher_outputs[4]
        teacher_reps = [teacher_rep for teacher_rep in teacher_reps]
        
    if hasattr(teacher_model, "module"):
        module_t = teacher_model.module
        module_s = student_model.module
    else:
        module_t = teacher_model
        module_s = student_model
        
    teacher_layer_num = module_t.config.num_hidden_layers
    student_layer_num = module_s.config.num_hidden_layers
    
    teacher_hidden_size = module_t.config.hidden_size
    student_hidden_size = module_s.config.hidden_size
    teacher_attention_num = module_t.config.num_attention_heads
    student_attention_num = module_s.config.num_attention_heads
    
    teacher_attention_size = teacher_hidden_size / teacher_attention_num
    student_attention_size = student_hidden_size / student_attention_num
    
    layers_per_block = int(teacher_layer_num / student_layer_num)
    teacher_atts = compute_relations(teacher_qs, teacher_ks, teacher_attention_size)
    student_atts = compute_relations(student_qs, student_ks, student_attention_size)
    
    new_teacher_atts = torch.stack(teacher_atts, dim=0)
    new_student_atts = torch.stack(student_atts, dim=0)
    new_teacher_reps = torch.stack(teacher_reps, dim=0)
    new_student_reps = torch.stack(student_reps, dim=0)
        
    if args.logit_distill:
        kd_fct.train()
        kd_loss = kd_fct(student_logits, teacher_logits, mode=output_mode)
        return [kd_loss]
    else:
        device = new_student_atts.device
        loss_mse = MSELoss()
        att_loss, rep_loss = \
            emd_fct(new_student_atts, new_teacher_atts, new_student_reps, new_teacher_reps,
                    device, loss_mse, args, T=args.T_emd)
        emb_loss = loss_mse(new_student_reps[0], new_teacher_reps[0])
        loss = att_loss + rep_loss + emb_loss
        return loss, att_loss, rep_loss, emb_loss
    
def tinybert_trains(args, teacher_model, student_model, input_ids, input_masks, segment_ids, 
                    label_ids, output_mode="classification", refine_model=None):
    
    def compute_relations(objs1, objs2, head_size):
        rels = []
        if type(objs1) == list:
            for obj1, obj2 in zip(objs1, objs2):
                rel = torch.matmul(obj1, obj2.transpose(-1, -2))
                rel = rel / math.sqrt(head_size)
                rels.append(rel)
        else:
            rel = torch.matmul(objs1, objs2.transpose(-1, -2))
            rel = rel / math.sqrt(head_size)
            rels.append(rel)
        return rels
    
    kd_fct = DistillKL(T=args.temperature)
    
    rep_fct = Hidden_MSE()
    att_fct = Attention_KL()
    
    loss = 0.
    
    if args.use_mixup:
        lam = np.random.beta(1.0, 1.0)
        lam = max(lam, 1-lam)
    else:
        lam = None
    
    student_outputs = student_model(input_ids, segment_ids, input_masks, 
                                    mixup=args.use_mixup, lam=lam, is_student=True)
    student_logits, student_qs, student_ks, student_vs, student_reps = \
        student_outputs[0], student_outputs[1], student_outputs[2], \
        student_outputs[3], student_outputs[4]
    
    if args.use_mixup and args.use_ic:
        with torch.no_grad():
            if refine_model is not None:
                student_outputs2 = refine_model.ema(input_ids, segment_ids, input_masks,
                                                mixup=False, is_student=True)
            else:
                student_outputs2 = student_model(input_ids, segment_ids, input_masks,
                                                 mixup=False, is_student=True)
            student_qs2, student_ks2, student_vs2, student_reps2 = \
                student_outputs2[1], student_outputs2[2], student_outputs2[3], student_outputs2[4]
    
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids, segment_ids, input_masks,
                                        mixup=args.use_mixup, lam=lam, is_student=False)
        teacher_logits, teacher_qs, teacher_ks, teacher_vs, teacher_reps = \
            teacher_outputs[0], teacher_outputs[1], teacher_outputs[2], \
            teacher_outputs[3], teacher_outputs[4]
        teacher_reps = [teacher_rep for teacher_rep in teacher_reps]
            
    if hasattr(teacher_model, "module"):
        module_t = teacher_model.module
        module_s = student_model.module
    else:
        module_t = teacher_model
        module_s = student_model
        
    teacher_layer_num = module_t.config.num_hidden_layers
    student_layer_num = module_s.config.num_hidden_layers
    
    teacher_hidden_size = module_t.config.hidden_size
    student_hidden_size = module_s.config.hidden_size
    teacher_attention_num = module_t.config.num_attention_heads
    student_attention_num = module_s.config.num_attention_heads
    
    teacher_attention_size = teacher_hidden_size / teacher_attention_num
    student_attention_size = student_hidden_size / student_attention_num
    
    assert teacher_layer_num % student_layer_num == 0
    layers_per_block = int(teacher_layer_num / student_layer_num)
    
    if args.logit_distill:
        kd_loss = kd_fct(student_logits, teacher_logits, mode=output_mode)
        return [kd_loss]
    
    if "A" in args.distill_object:
        teacher_atts = compute_relations(teacher_qs, teacher_ks, teacher_attention_size)
        student_atts = compute_relations(student_qs, student_ks, student_attention_size)
        if args.attention_l2l:
            new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]
            new_teacher_atts = torch.stack(new_teacher_atts, dim=0)
            new_student_atts = torch.stack(student_atts, dim=0)
        else:
            new_teacher_atts = teacher_atts[-1].unsqueeze(0)
            new_student_atts = student_atts[-1].unsqueeze(0)
        att_fct.train()
        loss += att_fct(new_student_atts, new_teacher_atts, input_masks)
        
        if args.use_mixup and args.use_ic_att:
            assert args.attention_l2l is not True
            student_atts2 = compute_relations(student_qs2, student_ks2, student_attention_size)
            new_student_atts2 = student_atts2[-1].unsqueeze(0)
            
            bsz = student_logits.size(0)
            rev = bsz - torch.arange(bsz) - 1
            loss += args.w_ic * (args.epoch / args.num_train_epochs) * lam * att_fct(new_student_atts, new_student_atts2, input_masks)
            loss += args.w_ic * (args.epoch / args.num_train_epochs) * (1 - lam) * att_fct(new_student_atts, new_student_atts2[:, rev], input_masks[rev])
        
    if "P" in args.distill_object:
        rep_fct = PKD_Loss(p=2, normalize=True)
        if args.hidden_l2l:
            new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)][1:]
            new_student_reps = [student_rep for student_rep in student_reps][1:]
            new_teacher_reps = torch.stack(new_teacher_reps, dim=1)[:,:,0,:]
            new_student_reps = torch.stack(new_student_reps, dim=1)[:,:,0,:]
        else:
            new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)][-1]
            new_student_reps = [student_rep for student_rep in student_reps][-1]
            new_teacher_reps = new_teacher_reps.unsqueeze(1)[:,:,0,:]
            new_student_reps = new_student_reps.unsqueeze(1)[:,:,0,:]
        rep_fct.train()
        loss += rep_fct(new_student_reps, new_teacher_reps)
        
    if "H" in args.distill_object:
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num+1)]
        if args.hidden_l2l:
            new_teacher_reps = torch.stack(new_teacher_reps, dim=0)
            new_student_reps = torch.stack(student_reps, dim=0)
        else:
            new_teacher_reps = new_teacher_reps[-1].unsqueeze(0)
            new_student_reps = student_reps[-1].unsqueeze(0)

        rep_fct.train()
        loss += rep_fct(new_student_reps, new_teacher_reps)
        
        if args.use_mixup and args.use_ic_rep:
            assert args.hidden_l2l is not True
            new_student_reps2 = student_reps2[-1].unsqueeze(0)
            
            bsz = student_logits.size(0)
            rev = bsz - torch.arange(bsz) - 1
            loss += args.w_ic * (args.epoch / args.num_train_epochs) * lam * rep_fct(new_student_reps, new_student_reps2)
            loss += args.w_ic * (args.epoch / args.num_train_epochs) * (1 - lam) * rep_fct(new_student_reps, new_student_reps2[:, rev])
        
    if "Q" in args.distill_object:
        teacher_qrys = compute_relations(teacher_qs, teacher_qs, teacher_attention_size)
        student_qrys = compute_relations(student_qs, student_qs, student_attention_size)

        if args.attention_l2l:
            new_teacher_qrys = [teacher_qrys[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]
            new_teacher_qrys = torch.stack(new_teacher_qrys, dim=0)
            new_student_qrys = torch.stack(student_qrys, dim=0)
        else:
            new_teacher_qrys = teacher_qrys[-1].unsqueeze(0)
            new_student_qrys = student_qrys[-1].unsqueeze(0)
            
        att_fct.train()
        loss += att_fct(new_student_qrys, new_teacher_qrys)

    if "K" in args.distill_object:
        teacher_keys = compute_relations(teacher_ks, teacher_ks, teacher_attention_size)
        student_keys = compute_relations(student_ks, student_ks, student_attention_size)

        if args.attention_l2l:
            new_teacher_keys = [teacher_keys[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]
            new_teacher_keys = torch.stack(new_teacher_keys, dim=0)
            new_student_keys = torch.stack(student_keys, dim=0)
        else:
            new_teacher_keys = teacher_keys[-1].unsqueeze(0)
            new_student_keys = student_keys[-1].unsqueeze(0)
            
        att_fct.train()
        loss += att_fct(new_student_keys, new_teacher_keys)

    if "V" in args.distill_object:
        teacher_vals = compute_relations(teacher_vs, teacher_vs, teacher_attention_size)
        student_vals = compute_relations(student_vs, student_vs, student_attention_size)

        if args.attention_l2l:
            new_teacher_vals = [teacher_vals[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]
            new_teacher_vals = torch.stack(new_teacher_vals, dim=0)
            new_student_vals = torch.stack(student_vals, dim=0)
        else:
            new_teacher_vals = teacher_vals[-1].unsqueeze(0)
            new_student_vals = student_vals[-1].unsqueeze(0)
            
        att_fct.train()
        loss += att_fct(new_student_vals, new_teacher_vals)
        
    if args.joint_train:
        kd_loss = kd_fct(student_logits, teacher_logits, mode=output_mode)
        loss += (1 - args.w_kd) * loss + args.w_kd * kd_loss
        
    return [loss]