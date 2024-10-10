import torch
import gin
import numpy as np

from torch import nn
import torch.nn.functional as F
from models.cka import linear_CKA, kernel_CKA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        # if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        # loss = -1 * (1-pt)**self.gamma * logpt
        loss = -1 * logpt#/ (pt + 1e-3)
        if self.size_average: return loss.mean()
        else: return loss.sum()

def cal_dist(inputs, inputs_center):
    n = inputs.size(0)
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # Compute pairwise distance, replace by the official when merged
    dist_center = torch.pow(inputs_center, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_center = dist_center + dist_center.t()
    dist_center.addmm_(1, -2, inputs_center, inputs_center.t())
    dist_center = dist_center.clamp(min=1e-12).sqrt()  # for numerical stability
    loss = torch.mean(torch.norm(dist-dist_center,p=2))
    return loss

def cal_dist_cosine(inputs, inputs_center):
    # We would like to perform cosine similarity for pairwise distance
    n = inputs.size(0)
    # Compute pairwise distance, replace by the official when merged
    dist = 1 - F.cosine_similarity(inputs.unsqueeze(1), inputs, dim=-1, eps=1e-30)
    dist = dist.clamp(min=1e-12)

    # Compute pairwise distance, replace by the official when merged
    dist_center = 1 - F.cosine_similarity(inputs_center.unsqueeze(1), inputs_center, dim=-1, eps=1e-30)
    dist_center = dist_center.clamp(min=1e-12)
    loss = torch.mean(torch.norm(dist-dist_center,p=2))
    return loss

def distillation_loss(fs, ft, opt='l2', delta=0.5,  reduce=True):
    if opt == 'l2':
        return (fs-ft).pow(2).sum(1).mean()
    if opt == 'l1':
        return (fs-ft).abs().sum(1).mean()
    if opt == 'huber':
        l1 = (fs-ft).abs()
        binary_mask_l1 = (l1.sum(1) > delta).type(torch.FloatTensor).unsqueeze(1).cuda()
        binary_mask_l2 = (l1.sum(1) <= delta).type(torch.FloatTensor).unsqueeze(1).cuda()
        loss = (l1.pow(2) * binary_mask_l2 * 0.5).sum(1) + (l1 * binary_mask_l1).sum(1) * delta - delta ** 2 * 0.5
        loss = loss.mean()
        return loss
    if opt == 'rkd':
        return cal_dist(fs, ft)
    if opt == 'cos':
        
        distance = 1 - F.cosine_similarity(fs, ft, dim=-1, eps=1e-30)
        if reduce:
            return distance.mean()
        else:
            return distance.reshape(-1,1)

    if opt == 'rkdcos':
        return cal_dist_cosine(fs, ft)
    if opt == 'linearcka':
        return 1 - linear_CKA(fs, ft)
    if opt == 'kernelcka':
        return 1 - kernel_CKA(fs, ft)


def cross_entropy_loss(logits, targets, temperature=1):
    T = 1
    log_p_y = F.log_softmax(logits/T, dim=1)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    acc = torch.eq(preds, labels).float().mean()
    stats_dict = {'loss': loss.item(), 'acc': acc.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}
    return loss, stats_dict, pred_dict

# logistic regression
def lr_loss(support_embeddings, support_labels,
            query_embeddings, query_labels, normalize=False):
    n_way = len(query_labels.unique())
    if normalize:
        support_embeddings = F.normalize(support_embeddings, dim=-1, p=2)
        query_embeddings = F.normalize(query_embeddings, dim=-1, p=2)
    support_embeddings = support_embeddings.detach().cpu().numpy()
    query_embeddings = query_embeddings.detach().cpu().numpy()
    support_labels = support_labels.view(-1).cpu().numpy()
    clf = LogisticRegression(penalty='none',
                             random_state=0,
                             C=1.0,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='multinomial')
    clf.fit(support_embeddings, support_labels)
    logits_ = clf.predict(query_embeddings)
    logits_ = torch.from_numpy(logits_).to(query_labels.device)
    logits = torch.zeros(query_labels.size(0), n_way).to(query_labels.device).scatter_(1, logits_.view(-1,1), 1) * 10

    return cross_entropy_loss(logits, query_labels)
    

# support vector machines
def svm_loss(support_embeddings, support_labels,
            query_embeddings, query_labels, normalize=False):
    n_way = len(query_labels.unique())
    if normalize:
        support_embeddings = F.normalize(support_embeddings, dim=-1, p=2)
        query_embeddings = F.normalize(query_embeddings, dim=-1, p=2)
    support_embeddings = support_embeddings.detach().cpu().numpy()
    query_embeddings = query_embeddings.detach().cpu().numpy()
    support_labels = support_labels.view(-1).cpu().numpy()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',
                                              C=1,
                                              kernel='linear',
                                              decision_function_shape='ovr'))
    clf.fit(support_embeddings, support_labels)
    logits_ = clf.predict(query_embeddings)
    logits_ = torch.from_numpy(logits_).to(query_labels.device)
    logits = torch.zeros(query_labels.size(0), n_way).to(query_labels.device).scatter_(1, logits_.view(-1,1), 1) * 10

    return cross_entropy_loss(logits, query_labels)

    
# NCC
def prototype_loss(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos'):
    n_way = len(query_labels.unique())

    prots = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)
    embeds = query_embeddings.unsqueeze(1)

    if distance == 'l2':
        logits = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', embeds, prots)
    elif distance == 'corr':
        logits = F.normalize((embeds * prots).sum(-1), dim=-1, p=2) * 10

    loss, stats_dict, pred_dict= cross_entropy_loss(logits, query_labels)
    # focal_loss = FocalLoss(gamma=0.5)
    # loss = focal_loss(logits, query_labels)
    return loss, stats_dict, pred_dict

# NCC - distill
def prototype_distill_loss(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos', kl_distill=None):
    n_way = len(query_labels.unique())

    prots = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)
    query_prots = compute_prototypes(query_embeddings, query_labels, n_way).unsqueeze(0)
    support_embeds = support_embeddings.unsqueeze(1)
    query_embeds = query_embeddings.unsqueeze(1)

    if distance == 'l2':
        support_logits = -torch.pow(support_embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
        query_logits = -torch.pow(query_embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        support_logits = F.cosine_similarity(support_embeds, prots, dim=-1, eps=1e-30) * 10
        query_logits = F.cosine_similarity(query_embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        support_logits = torch.einsum('izd,zjd->ij', support_embeds, prots)
        query_logits = torch.einsum('izd,zjd->ij', query_embeds, prots)
    elif distance == 'corr':
        support_logits = F.normalize((support_embeds * prots).sum(-1), dim=-1, p=2) * 10
        query_logits = F.normalize((query_embeds * prots).sum(-1), dim=-1, p=2) * 10

    loss = kl_distill(support_logits, query_logits)

    return loss

def compute_prototypes(embeddings, labels, n_way):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = embeddings[(labels == i).nonzero(as_tuple=False), :].mean(0)
    return prots

# compute similarity between features
def compute_sim(embeddings, labels, n_way):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    intra_sim = torch.zeros(n_way).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        x = embeddings[(labels == i).nonzero(), :]
        prots[i] = x.mean(0)
        intra = F.cosine_similarity(x, x.squeeze(1), dim=-1, eps=1e-30).triu(diagonal=-1)
        if (intra!=0).sum()!=0:
            intra_sim[i] = intra.sum()/(intra!=0).sum()
        else:
            intra_sim[i] = intra.sum()
        
    inter_sim = F.cosine_similarity(prots.unsqueeze(1), prots, dim=-1, eps=1e-30).triu(diagonal=-1)
    inter_sim = inter_sim.sum()/(inter_sim!=0).sum()
    whole_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings, dim=-1, eps=1e-30).triu(diagonal=-1)
    whole_sim = whole_sim.sum()/(whole_sim!=0).sum()
    return (intra_sim.mean(), inter_sim, whole_sim)

# knn
def knn_loss(support_embeddings, support_labels,
             query_embeddings, query_labels, distance='cos'):
    n_way = len(query_labels.unique())

    prots = support_embeddings
    embeds = query_embeddings.unsqueeze(1)

    if distance == 'l2':
        dist = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        dist = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        dist = torch.einsum('izd,zjd->ij', embeds, prots)
    _, inds = torch.topk(dist, k=1)

    logits = torch.zeros(embeds.size(0), n_way).to(embeds.device).scatter_(1, support_labels[inds.flatten()].view(-1,1), 1) * 10

    return cross_entropy_loss(logits, query_labels)


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss


class AdaptiveCosineNCC(nn.Module):
    def __init__(self):
        super(AdaptiveCosineNCC, self).__init__()
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)

    def forward(self, support_embeddings, support_labels,
                query_embeddings, query_labels, return_logits=False):
        n_way = len(query_labels.unique())

        prots, _  = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)
        embeds = query_embeddings.unsqueeze(1)
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * self.scale

        if return_logits:
            return logits

        return cross_entropy_loss(logits, query_labels)



