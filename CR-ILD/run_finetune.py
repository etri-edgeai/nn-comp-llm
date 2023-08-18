"""BERT finetuning runner."""
from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                                TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from modules.modeling import TinyBertForSequenceClassification
from modules.tokenization import BertTokenizer
from modules.optimization import BertAdam
from modules.file_utils import WEIGHTS_NAME, CONFIG_NAME
from modules.ema import ModelEMA

from dataloaders import get_dataloader
from training_functions import mlm_trains, tinybert_trains
from training_functions import emdbert_trains, pkd_trains, kd_trains
from utils import pkd_initialization
from utils import classifier_reuse_initialization
from losses import EMD_Loss
from opts import parse_args

csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1)/2
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            
def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    model.eval()
    
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            outputs = model(input_ids, segment_ids, input_mask)
            logits = outputs[0]
            
        # create eval loss and other metric required by the task

        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
            
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            
    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    
    return result

def main():    
    args = parse_args()
    logger.info('The args: {}'.format(args))

    if args.use_ic_att or args.use_ic_rep:
        args.use_ic = True
    else:
        args.use_ic = False
    
    if "TinyBERT" in args.mode:
        if "-" in args.mode:
            args.distill_object = args.mode.split("-")[1]
        elif args.mode == "TinyBERT":
            args.distill_object += "AH"
        else:
            raise NotImplementedError
            
    if "MiniLM" in args.mode:
        if "-" in args.mode:
            args.distill_object = args.mode.split("-")[1]
        elif args.mode == "MiniLM":
            args.distill_object = "AV"
        elif args.mode == "MiniLM2":
            args.distill_object = "QKV"
        else:
            raise NotImplementedError
            
    if "CID" in args.mode:
        if "-" in args.mode:
            args.distill_object = args.mode.split("-")[1]
        elif args.mode == "CID":
            args.distill_object += "AH"
        else:
            raise NotImplementedError
            
    if args.mode == "EMD-A":
        args.mode = "EMD"
        args.use_att = True
        args.use_rep = False
        args.seperate = True
    elif args.mode == "EMD-H":
        args.mode = "EMD"
        args.use_att = False
        args.use_rep = True
        args.seperate = True
    
    default_params = {
        "cola": {"num_train_epochs": 20, "max_seq_length": 64},
        "mnli": {"num_train_epochs": 5, "max_seq_length": 128},
        "mrpc": {"num_train_epochs": 10, "max_seq_length": 128},
        "sst-2": {"num_train_epochs": 10, "max_seq_length": 64},
        "sts-b": {"num_train_epochs": 20, "max_seq_length": 128},
        "qqp": {"num_train_epochs": 5, "max_seq_length": 128},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
        "rte": {"num_train_epochs": 10, "max_seq_length": 128},
        "wiki": {"num_train_epochs": 20, "max_seq_length": 128}
    }
    
    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]
    
    assert (torch.cuda.is_available())
    
    # apex 
    if args.local_rank == -1:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuid)
        device = torch.device("cuda")
        args.rank = 0
    else:
        raise NotImplementedError
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {}".format(device))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    task_name = args.task_name.lower()
    if task_name in default_params:
        args.max_seq_len = default_params[task_name]["max_seq_length"]
        if "TinyBERT" in args.mode or "EMD" in args.mode:
            args.num_train_epochs = default_params[task_name]["num_train_epochs"]
    
    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)
    train_dataloader, eval_dataloader, eval_labels = get_dataloader(args, tokenizer, logger)
    
    teacher_model = None
    if args.mode!="FT":
        teacher_model = TinyBertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=args.num_labels)
        teacher_model.to(device)
        teacher_model.eval()
        args.teacher_config = teacher_model.config
        
    if not args.scratch:
        student_model = TinyBertForSequenceClassification.from_pretrained(args.student_model, num_labels=args.num_labels)
    else:
        student_model = TinyBertForSequenceClassification.from_scratch(args.student_model, num_labels=args.num_labels)
        
    if args.pkd_initialize and teacher_model:
        base_teacher = './pretrain/bert_base'
        print (base_teacher)
        teacher_model_ = TinyBertForSequenceClassification.from_pretrained(base_teacher, num_labels=args.num_labels)
        student_model = pkd_initialization(teacher_model_, student_model)
        
    student_model.to(device)
    args.student_config = student_model.config
    
    if "TinyBERT" in args.mode:
        if args.use_ema:
            refine_model = ModelEMA(args, student_model, device)
        else:
            refine_model = None
    
    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataloader.dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval(student_model, task_name, eval_dataloader,
                         device, args.output_mode, eval_labels, args.num_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader.dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.num_train_optimization_steps)
        
        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()
            
        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                             schedule="warmup_linear",
                             t_total=args.num_train_optimization_steps,
                             warmup=args.warmup_proportion)
        
        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0
        if args.mode in ["KD", "PKD"]:
            output_eval_file = os.path.join(args.output_dir, "eval_results_kd.txt")
        elif args.mode == "FT":
            output_eval_file = os.path.join(args.output_dir, "eval_results_ft.txt")
        else:
            outputs_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        
        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            args.epoch = epoch_
            tr_loss = 0.0
            tr_acc = 0.0
            
            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            
            if args.mode == "EMD":
                transformer_loss = EMD_Loss(args)
            
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)
                
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue
                    
                if "TinyBERT" in args.mode:
                    outputs = tinybert_trains(args, teacher_model, student_model,
                                              input_ids, input_mask, segment_ids,
                                              label_ids, output_mode=args.output_mode,
                                              refine_model=refine_model)
                    loss = outputs[0]
                    
                elif "MiniLM" in args.mode:
                    outputs = minilm_trains(args, teacher_model, student_model,
                                            input_ids, input_mask, segment_ids,
                                            label_ids, output_mode=args.output_mode)
                    loss = outputs[0]
                    
                elif "EMD" in args.mode:
                    outputs = emdbert_trains(args, teacher_model, student_model,
                                             input_ids, input_mask, segment_ids,
                                             label_ids, transformer_loss, args.output_mode)
                    loss = outputs[0]
                
                elif args.mode in ["LTI_P"]:
                    outputs = pkd_trains(args, teacher_model, student_model,
                                         input_ids, input_mask, segment_ids,
                                         label_ids, output_mode=args.output_mode)
                    loss = outputs[0]
                    
                elif args.mode in ["KD", "MixKD", "PKD", "MixCons"]:
                    outputs = kd_trains(args, teacher_model, student_model,
                                        input_ids, input_mask, segment_ids,
                                        label_ids, output_mode=args.output_mode)
                    loss = outputs[0]
                    
                elif args.mode == "FT":
                    student_outputs = student_model(input_ids, segment_ids, input_mask,
                                                    is_student=True)
                    logits = student_outputs[0]
                        
                    if args.output_mode == "classification":
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits, label_ids)
                        pred_cls = logits.data.max(1)[1]
                        tr_acc += pred_cls.eq(label_ids).sum().cpu().item()
                    elif args.output_mode == "regression":
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), label_ids.view(-1))
                        
                tr_loss += loss.item()
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
                    
                    optimizer.step()
                    if args.use_ema:
                        refine_model.update(student_model)
                    optimizer.zero_grad()
                    global_step += 1
                    
                if (global_step + 1) % args.eval_step == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_dataloader.dataset))
                    logger.info("  Batch size = %d", args.eval_batch_size)
                    
                    student_model.eval()
                    loss = tr_loss / (step + 1)
                    
                    results = {}
                    save_model = False
                    
                    if ("TinyBERT" in args.mode or "MiniLM" in args.mode) and not args.logit_distill and not args.joint_train:
                        continue
                            
                    else:
                        result = do_eval(student_model, task_name, eval_dataloader,
                                         device, args.output_mode, eval_labels, args.num_labels)
                        result['global_step'] = global_step
                        result['loss'] = loss
                        result['tr_acc'] = tr_acc / nb_tr_examples
                        result_to_file(result, output_eval_file)

                        if task_name in acc_tasks and result['acc'] > best_dev_acc:
                            best_dev_acc = result['acc']
                            save_model = True

                        if task_name in corr_tasks and result['corr'] > best_dev_acc:
                            best_dev_acc = result['corr']
                            save_model = True

                        if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                            best_dev_acc = result['mcc']
                            save_model = True
                        
                    if save_model and args.rank == 0:
                        logger.info("***** Save model *****")
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                        model_name = WEIGHTS_NAME
                        
                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                        
                        if args.save_model:
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)
                    student_model.train()
                    
        if args.rank == 0 and args.save_final_model:
            logger.info("***** Save model *****")
            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
            model_name = WEIGHTS_NAME

            output_model_file = os.path.join(args.output_dir, model_name)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)

if __name__ == "__main__":
    main()