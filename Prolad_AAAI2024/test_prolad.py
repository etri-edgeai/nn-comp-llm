"""
This code allows you to evaluate performance of a single feature extractor + test with TAD(our methods)
"""

import os
import torch
import tensorflow as tf
tf.compat.v1.disable_eager_execution() #added part
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from utils import check_dir

from models.losses import prototype_loss, knn_loss, lr_loss
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from models.prolad import resnet_prolad_plus, prolad_plus
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES)
from config import args
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
from scipy.stats import entropy,wasserstein_distance
import pandas as pd
import wandb

# All Datasets : omniglot vgg_flower traffic_sign mnist fungi quickdraw cu_birds cifar10 mscoco dtd cifar100 aircraft ilsvrc_2012
def main():
    TEST_SIZE = args['test.size']

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    if args['data.test'] == 'All':
        testsets = ALL_METADATASET_NAMES # comment this line to test the model on args['data.test']
    if args['test.mode'] == 'mdl':
        # multi-domain learning setting, meta-train on 8 training sets
        trainsets = TRAIN_METADATASET_NAMES
    elif args['test.mode'] == 'sdl':
        # single-domain learning setting, meta-train on ImageNet
        trainsets = ['ilsvrc_2012']
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])
    model = get_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model.eval()
    model = resnet_prolad_plus(model)
    model.reset()
    model.cuda()

    accs_names = ['NCC']
    var_accs = dict()
    coeffs = dict()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False

    model_dataset_pairs = []
    feature_dataset_pairs = []
    is_config=False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            if dataset in trainsets:
                scale = 0.1
            else:
                scale = 1.0
            print(dataset)
            var_accs[dataset] = {name: [] for name in accs_names}
            coeffs[dataset] = []

            for i in tqdm(range(TEST_SIZE)):
                # initialize adapters and pre-classifier alignment for each task
                model.reset()
                # loading a task containing a support set and a query set
                sample = test_loader.get_test_task(session, dataset)
                context_images = sample['context_images']
                context_labels = sample['context_labels']
        

                # optimize adapters
                used_lr, used_lr_beta, used_scale,eff, momentum = prolad_plus(context_images, context_labels, model, max_iter=40, scale=scale, distance=args['test.distance'])
                if type(eff[1]) == torch.Tensor:
                    coeffs[dataset].append(eff[1].item())
                else:
                    coeffs[dataset].append(eff[1])
                
                # evaluate adapters on query set
                target_labels = sample['target_labels']
                with torch.no_grad():
                    context_features = model.embed_concat(sample['context_images'])
                    target_features = model.embed_concat(sample['target_images'])
                   
                _, stats_dict, _ = prototype_loss(
                    context_features, context_labels,
                    target_features, target_labels, distance=args['test.distance'])

                var_accs[dataset]['NCC'].append(stats_dict['acc'])
                if ((i+1)%20 == 0):
                    d_acc = np.array(var_accs[dataset]['NCC']) * 100
                    print(f"{dataset}: current (lr: {used_lr}, scale: {used_scale})(len: ({len(d_acc)})), eff: {eff} , momentum: {momentum}, test_acc {d_acc.mean():.2f}%")
                    
            dataset_acc = np.array(var_accs[dataset]['NCC']) * 100
            print(f"{dataset}: test_acc {dataset_acc.mean():.2f}%")
            avg_coeff = np.mean(coeffs[dataset], axis=0)
    for dataset in testsets:
        avg_coeff = np.mean(coeffs[dataset], axis=0)
        print(f"{dataset}: avg_coeff {avg_coeff}")
    
    # Print nice results table
    print('results of {} with {}'.format(args['model.name'], args['test.prolad_opt']))
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
    out_path = os.path.join(args['out.dir'], args['out.method'])
    out_path = check_dir(out_path, True)
    out_path_txt = os.path.join(out_path, 'prolad_output.txt')
    out_path = os.path.join(out_path, 'prolad_output.npy')
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")
    with open(out_path_txt, 'w') as f:
        f.write(table)


if __name__ == '__main__':
    main()



