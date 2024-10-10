'''
prolad.py
Main logic of our project. We first attach adatpers by each block. Then we first train TA, which devoids of normalization layer and set the teacher features. Finally, train both TA and TAN with progressive learning and adaptive distillation. 
'''

import torch
import torch.nn as nn
import numpy as np
import gc
from config import args
import copy
import torch.nn.functional as F
from models.losses import prototype_loss, distillation_loss, compute_sim, DistillKL, prototype_distill_loss
from utils import device
import torchvision
import matplotlib.pyplot as plt


# momentum size for TAN
Momentum = 0.8


# initialization functions 
def init_batch(m):
    if type(m) == nn.BatchNorm2d and m.momentum == Momentum: 
        m.reset_running_stats()

def train_batch(m):
    if type(m) == nn.BatchNorm2d and m.momentum == Momentum:
        m.train()

def mode_ta(m):
    if hasattr(m,'mode'):
        m.mode = 'ta'

def mode_tan(m):
    if hasattr(m,'mode'):
        m.mode = 'tan'

class BasicBlock_prolad(nn.Module):
    """
        Adapters are attached to each block
        Each mode refers to different adapter. 
    """
    def __init__(self, block):
        super(BasicBlock_prolad, self).__init__()
        self.block = copy.deepcopy(block)
        self.conv1 = conv_prolad_up(self.block.conv1)
        planes, in_planes, kernel_size, _ = self.block.conv1.weight.size()
        self.in_planes = in_planes
        self.planes = planes
        self.bn1 =  ( self.block.bn1 ) 
        self.relu = self.block.relu  
        self.conv2 = conv_prolad_up( self.block.conv2 )  
        self.bn2 = (  self.block.bn2 )  
        self.downsample = self.block.downsample  
        self.stride = self.block.stride
        self.padding = self.block.conv2.padding
        self.mode = 'tan'

        self.groups = 8


        self.alpha_norm = nn.Parameter(torch.ones(planes, planes//self.groups, kernel_size, kernel_size), requires_grad=True)
        # self.alpha_norm = nn.Parameter(torch.ones(planes, planes, 1, 1), requires_grad=True)
        # self.alpha_weight = nn.Parameter(torch.ones(1, planes, 1, 1), requires_grad=True)
        # self.alpha_bias = nn.Parameter(torch.ones(1, planes, 1, 1), requires_grad=True)

    
        self.a_bn = nn.BatchNorm2d(planes, eps=1e-07, momentum=Momentum, affine=False, track_running_stats=True)
        self.avgpool = nn.AvgPool2d(3, padding=1, stride=1)
        

        self.relu = nn.ReLU() 

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out = self.bn1(out1)

        out2 = self.relu(out)         
        out2 = self.conv2(out2) 
  
        out = self.bn2(out2) 

    
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.mode == 'tan':
            '''
                adapter TAN that incorporates normalization layer
            '''
            identity = identity + (F.conv2d(self.a_bn(out1), self.alpha_norm, padding=1, groups=self.groups)) 

        out = out + identity  

        # out = self.a_dropout(out)
        
        out = self.relu(out)

    

        return out



class conv_prolad_up(nn.Module):
    '''
        attach matrix adapters at each 3x3 convolutional layer
    '''
    def __init__(self, orig_conv):
        super(conv_prolad_up, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, self.kernel_size, kernel_size = self.conv.weight.size()
        self.in_planes = in_planes
        self.mode = 'prolad'
        # task-specific adapters
        if 'alpha' in args['test.prolad_opt']:
            
            self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1), requires_grad=True)

            self.avgpool = nn.AvgPool2d(kernel_size, padding=self.conv.padding, stride=self.conv.stride)


                
    def forward(self, x):
        y = self.conv(x)
   
        if 'beta' in args['test.prolad_opt']:
            if self.mode=='ta':
                y = y + F.conv2d(x, self.alpha, stride=self.conv.stride) 
            elif self.mode=='prolad':
                y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)


                     
        return y


class pa(nn.Module):
    """ 
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    In our paper, it is denoted as c
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim),    requires_grad = True)

        self.feat_dim = feat_dim
        
        self.layer_norm = nn.LayerNorm(feat_dim, elementwise_affine=False)


    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)

        else:
            x = x.flatten(1)
            x = x.unsqueeze(-1).unsqueeze(-1)

        x = (F.conv2d(x, self.weight.to(x.device))).flatten(1)


        return x

class resnet_prolad_plus(nn.Module):
    """ Attaching adapters to the ResNet backbone """
    def __init__(self, orig_resnet):
        super(resnet_prolad_plus, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
            v.requires_grad=False

        # copy the original resnet for calcualting the adaptive coefficient
        self.orig_resnet = copy.deepcopy(orig_resnet)

        # self.alpha_weight_param = nn.Parameter(torch.ones(1), requires_grad=True)


        for i, block in enumerate(orig_resnet.layer1):
            orig_resnet.layer1[i] = BasicBlock_prolad(block)
        for i, block in enumerate(orig_resnet.layer2):
            orig_resnet.layer2[i] = BasicBlock_prolad(block)
        for i, block in enumerate(orig_resnet.layer3):
            orig_resnet.layer3[i] = BasicBlock_prolad(block)
        for i, block in enumerate(orig_resnet.layer4):
            orig_resnet.layer4[i] = BasicBlock_prolad(block)
        

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.layer4[-1].bn2.num_features
        beta_ta = pa(feat_dim)
        beta_orig = pa(feat_dim)
        beta_tan = pa(feat_dim)
        setattr(self, 'beta_ta', beta_ta)
        setattr(self, 'beta_tan', beta_tan)
        setattr(self, 'beta_orig', beta_orig)

    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x)

        
    def embed_concat(self, x, c_features=None):
        """
            Final forward propagation for evaluate on query set 
        """
        with torch.no_grad():

            self.apply(mode_tan)
            context_features_prolad = self.embed(x)
        
            
        if 'beta' in args['test.prolad_opt']:
            with torch.no_grad():

                aligned_features_prolad = self.beta_tan(context_features_prolad)

                t_features = aligned_features_prolad
                
        else:
            t_features = context_features_prolad
        
        return t_features

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self, scale=1.0):
        # initialize task-specific adapters (alpha)
        
        for k, v in self.named_parameters():
            if 'alpha' in k and v.requires_grad:
                # nn.init.constant_(v, 0.0)
                if 'bias' not in k:
                    if 'eff' in k:
                        v.data = torch.ones(v.size()).to(v.device)#*1e-4
                    elif 'weight' in k:
                        v.data = torch.ones(v.size()).to(v.device)*0.5

                      
                    else:
                        if 'norm' in k:
                            v.data = (torch.randn(v.size()).to(v.device))*0.0001

                        else:
                            v.data =  torch.randn(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)*0.00001

                else:
                    v.data = torch.zeros(v.size()).to(v.device)

            elif 'beta' in k and v.requires_grad:

                # initialize pre-classifier alignment mapping (beta)
                if 'bias' in k:
                    v.data = torch.zeros(v.size()).to(v.device)
                else:
                    v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
        self.apply(init_batch)



def train_adapter(optimizer, model, x, y, prolad_opt, max_iter, distance, beta='tan', t_features = None, a_coeff=0.0, scale=1.0, fixed=False):
    """
        Training selected adapters from scratch
    """
    
    it = 100
    a=0
    if fixed:
        it = max_iter
    for i in range(it):
            

        optimizer.zero_grad()
        model.zero_grad()

        if 'alpha' in prolad_opt and beta != 'orig':
            context_features = model.embed(x)

        if beta == 'ta':
            aligned_features = model.beta_ta(context_features)
        elif beta=='tan':
            aligned_features = model.beta_tan(context_features)
        elif beta=='orig':
            aligned_features = model.beta_orig(x)
        else:
            aligned_features = context_features

        

            
        loss, stat, _ = prototype_loss(aligned_features, y,
                                       aligned_features, y, distance=distance)

        loss = loss * scale

       
        if t_features != None:
            distill_loss = distillation_loss(aligned_features, t_features, opt=distance) 
            
            loss = (1-a_coeff)*loss + (a_coeff)* distill_loss 
            
      
        
        loss.backward()
        optimizer.step()


        """
            The number of epochs depend on accuracy 
            If Accuracy exceeds 99%, the process is finished with 15 more iterations 
        """
        if stat['acc'] > 0.99 and not fixed:
            a += 1
            if a > max_iter:
                break

def train_one_set(model, max_iter, lr, lr_w, lr_beta, prolad_opt, x, y, distance, beta='tan', reset=False, t_features = None, a_coeff=0.0, scale=1.0, fixed=False):
    """
        Select training process depending on the defined options

    """

    alpha_params = [v for k, v in model.named_parameters() if ('alpha' in k and ('alpha_weight' not in k and 'alpha_bias' not in k))]
    alpha_params_w = [v for k, v in model.named_parameters() if ('alpha' in k and ('alpha_weight' in k or 'alpha_bias' in k)) or 'a_bn' in k]


    beta_params = [v for k, v in model.named_parameters() if 'beta' in k]
    params = []
    
     
    if 'alpha' in prolad_opt:
        params.append({'params': alpha_params})
        params.append({'params': alpha_params_w, 'lr': lr_w})
    if 'beta' in prolad_opt:
        params.append({'params': beta_params, 'lr': lr_beta})
    
    optimizer = torch.optim.Adadelta(params, lr=lr)

    # linear probing
    if beta == 'orig':
        train_adapter(optimizer, model, x, y, prolad_opt, max_iter, distance, beta=beta, t_features = t_features, a_coeff=a_coeff, scale=scale, fixed=fixed)
        with torch.no_grad():
            aligned_features_t_o = model.beta_orig(x)
    # TAN + TA
    elif beta=='tan':
        model.apply(mode_tan)
        train_adapter(optimizer, model, x, y, prolad_opt, max_iter, distance, beta=beta, t_features = t_features, a_coeff= a_coeff, scale=scale, fixed=fixed)
        with torch.no_grad():
            aligned_features_t_o = model.beta_tan(model.embed(x))

    # TA    
    elif beta=='ta':
        model.apply(mode_ta)
        train_adapter(optimizer, model, x, y, prolad_opt, max_iter, distance, beta=beta, t_features = t_features, a_coeff=a_coeff, scale=scale, fixed=fixed)
        with torch.no_grad():
            aligned_features_t_o = model.beta_ta(model.embed(x))

    # to reset the parameters for space efficiency
    if reset:
        model.reset()

    # return teacher features 
    return aligned_features_t_o



def prolad_plus(context_images, context_labels, model, max_iter=40, scale=0.1, distance='cos'):
    """
    Optimizing adapters attached to the ResNet backbone, 
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """
    t_features = None

    model.eval()
    model.apply(train_batch)


    # caculate the statistics to compute adaptive coefficient
    with torch.no_grad():
        t_features = model.orig_resnet.embed(context_images)
        loss_t_o, stat_t_o, _ = prototype_loss(t_features, context_labels,
                                       t_features, context_labels, distance=distance)
        f_original = t_features
        acc_stat = stat_t_o['acc']
    

    prolad_opt = args['test.prolad_opt']

    lr_w = 5.0*scale
    
    lr = 0.5*scale
    lr_beta = 2*lr

    n_way = len(context_labels.unique())


    betas = ['ta']
    class_intra_sim, class_inter_sim, features_sim = compute_sim(t_features, context_labels, n_way)

    # ProLAD-sim
    eff_bias = 1.5

    a_coeff = min(max(torch.exp(-loss_t_o*(1-(class_inter_sim-features_sim)*1.5)**2),0.0), 1.0)

    # ProLAD-loss
    # eff_bias = 1.5
    # acc_stat = torch.Tensor([acc_stat, acc_stat])
    # acc_stat = torch.mean(acc_stat).to(context_images.device)
    # a_coeff = min(max(torch.exp(-loss_t_o*(1-acc_stat)*eff_bias),0.0), 1.0)

    
    lrs = torch.Tensor([lr, lr, lr])
    lr_betas = torch.Tensor([lr_beta, lr_beta, lr_beta])

     
    for i in range(len(betas)):
        x = context_images
        
        if betas[i] == 'orig':
            # for linear probing
            x = t_features


        ta_features = train_one_set(model, max_iter=25, lr=lrs[i], lr_w=lr_w, lr_beta=lr_betas[i], prolad_opt=prolad_opt, x=x, y=context_labels, distance=distance, beta=betas[i], reset=False, t_features = None, a_coeff=0.0, scale=1.0, fixed=False)
       

        t_features = ta_features



    lr = lr 
    loss_scale = 1.5
    # to train the adapters with distillation
    train_one_set(model, max_iter=25, lr=lr, lr_w=lr_w, lr_beta=lr_beta, prolad_opt=prolad_opt, x=context_images, y=context_labels, distance=distance, beta='tan', reset=False, t_features = t_features, a_coeff = a_coeff, scale=loss_scale, fixed=False)


    model.eval()

    # to save the coefficient
    a_coeff = a_coeff.to("cpu").detach().item()

    return (lr, lr_beta, loss_scale,(eff_bias,a_coeff), Momentum)

