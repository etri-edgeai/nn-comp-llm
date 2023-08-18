import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pyemd import emd_with_flow

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
        
    def forward(self, y_s, y_t, mode="classification"):
        if mode == "regression":
            loss = F.mse_loss((y_s/self.T).view(-1), (y_t/self.T).view(-1))
        else:
            p_s = F.log_softmax(y_s/self.T, dim=-1)
            p_t = F.softmax(y_t/self.T, dim=-1)
            loss = -torch.sum(p_t * p_s, dim=-1).mean() * (self.T ** 2)
        return loss
    
class PKD_Loss(nn.Module):
    def __init__(self, p, normalize=False):
        super(PKD_Loss, self).__init__()
        self.p = p
        self.normalize = normalize
        
    def forward(self, teacher_patience, student_patience):
        if self.normalize:
            if len(teacher_patience.size()) == 4:
                teacher_patience = F.normalize(teacher_patience, p=self.p, dim=3)
                student_patience = F.normalize(student_patience, p=self.p, dim=3)
            elif len(teacher_patience.size()) == 3:
                teacher_patience = F.normalize(teacher_patience, p=self.p, dim=2)
                student_patience = F.normalize(student_patience, p=self.p, dim=2)
        return F.mse_loss(teacher_patience.float(), student_patience.float())

class Hidden_MSE(nn.Module):
    def __init__(self):
        super(Hidden_MSE, self).__init__()
        
    def forward(self, student_reps, teacher_reps, index=None):
        rep_loss = 0.0
        layer_num, bsz, max_len, h_dim = student_reps.size()
        
        if index is not None:
            student_reps = student_reps[:, index]
            teacher_reps = teacher_reps[:, index]
        
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            student_rep = student_rep.view(-1, max_len, h_dim)
            teacher_rep = teacher_rep.view(-1, max_len, h_dim)
            if index is not None:
                rep_loss += F.mse_loss(student_rep, teacher_rep, reduction="mean") * bsz / len(index)
            else:
                rep_loss += F.mse_loss(student_rep, teacher_rep, reduction="mean")
        return rep_loss
    
class Hidden_MAE(nn.Module):
    def __init__(self):
        super(Hidden_MAE, self).__init__()
        
    def forward(self, student_reps, teacher_reps):
        rep_loss = 0.0
        layer_num, bsz, max_len, h_dim = student_reps.size()
        
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            student_rep = student_rep.view(-1, max_len, h_dim)
            teacher_rep = teacher_rep.view(-1, max_len, h_dim)
            rep_loss += F.l1_loss(student_rep, teacher_rep, reduction="mean")
        return rep_loss
        
class Attention_MSE(nn.Module):
    def __init_(self):
        super(Attention_MSE, self).__init__()
        
    def forward(self, student_atts, teacher_atts, attention_mask=None, index=None):
        att_loss = 0.0
        layer_num, bsz, head, max_len, max_len = student_atts.size()
        
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=student_atts.dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        for student_att, teacher_att in zip(student_atts, teacher_atts):
            if attention_mask is not None:
                student_att = student_att + extended_attention_mask
                teacher_att = teacher_att + extended_attention_mask
                
            if index is not None:
                student_att = student_att[index]
                teacher_att = teacher_att[index]
            
            student_att = student_att.view(-1, max_len, max_len)
            teacher_att = teacher_att.view(-1, max_len, max_len)
            
            student_att = torch.where(student_att <= -1e2, 
                                      torch.zeros_like(student_att).to(student_atts.device),
                                      student_att)
            teacher_att = torch.where(teacher_att <= -1e2, 
                                      torch.zeros_like(teacher_att).to(student_atts.device),
                                      teacher_att)
            if index is not None:
                att_loss += F.mse_loss(student_att, teacher_att, reduction="mean") * bsz / len(index)
            else:
                att_loss += F.mse_loss(student_att, teacher_att, reduction="mean")
        return att_loss
    
class Attention_MAE(nn.Module):
    def __init_(self):
        super(Attention_MAE, self).__init__()
        
    def forward(self, student_atts, teacher_atts, attention_mask=None):
        att_loss = 0.0
        layer_num, bsz, head, max_len, max_len = student_atts.size()
        
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=student_atts.dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        for student_att, teacher_att in zip(student_atts, teacher_atts):
            if attention_mask is not None:
                student_att = student_att + extended_attention_mask
                teacher_att = teacher_att + extended_attention_mask
            
            student_att = student_att.view(-1, max_len, max_len)
            teacher_att = teacher_att.view(-1, max_len, max_len)
            
            student_att = torch.where(student_att <= -1e2, 
                                      torch.zeros_like(student_att).to(student_atts.device),
                                      student_att)
            teacher_att = torch.where(teacher_att <= -1e2, 
                                      torch.zeros_like(teacher_att).to(student_atts.device),
                                      teacher_att)
            att_loss += F.l1_loss(student_att, teacher_att, reduction="mean")
        return att_loss
    
class Attention_KL(nn.Module):
    def __init__(self):
        super(Attention_KL, self).__init__()
        
    def forward(self, student_atts, teacher_atts, attention_mask=None):
        att_loss = 0.0
        layer_num, bsz, head, max_len, max_len = student_atts.size()
        
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=student_atts.dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        for student_att, teacher_att in zip(student_atts, teacher_atts):
            if attention_mask is not None:
                student_att = student_att + extended_attention_mask
                teacher_att = teacher_att + extended_attention_mask
            
            student_prob = F.log_softmax(student_att.view(-1, max_len), dim=-1)
            teacher_prob = F.softmax(teacher_att.view(-1, max_len), dim=-1)
            att_loss += -torch.mean(torch.sum(teacher_prob * student_prob, dim=-1))
        return att_loss
    
class EMD_Loss(nn.Module):
    def __init__(self, args):
        super(EMD_Loss, self).__init__()
        config = args.student_config
        self.att_student_weight = np.ones(config.num_hidden_layers) / config.num_hidden_layers
        self.rep_student_weight = np.ones(config.num_hidden_layers) / config.num_hidden_layers
        
        config = args.teacher_config
        self.att_teacher_weight = np.ones(config.num_hidden_layers) / config.num_hidden_layers
        self.rep_teacher_weight = np.ones(config.num_hidden_layers) / config.num_hidden_layers
        
    def get_new_layer_weight(self, trans_matrix, distance_matrix, 
                            stu_layer_num, tea_layer_num, 
                            T, type_update='att'):
        if type_update == 'att':
            student_layer_weight = np.copy(self.att_student_weight)
            teacher_layer_weight = np.copy(self.att_teacher_weight)
        else:
            student_layer_weight = np.copy(self.rep_student_weight)
            teacher_layer_weight = np.copy(self.rep_teacher_weight)
            
        distance_matrix = distance_matrix.detach().cpu().numpy().astype('float64')
        trans_weight = np.sum(trans_matrix * distance_matrix, -1)
        for i in range(stu_layer_num):
            student_layer_weight[i] = trans_weight[i] / student_layer_weight[i]
        weight_sum = np.sum(student_layer_weight)
        for i in range(stu_layer_num):
            if student_layer_weight[i] != 0:
                student_layer_weight[i] = weight_sum / student_layer_weight[i]
                
        trans_weight = np.sum(np.transpose(trans_matrix) * distance_matrix, -1)
        for j in range(tea_layer_num):
            teacher_layer_weight[j] = trans_weight[j + stu_layer_num] / teacher_layer_weight[j]
        weight_sum = np.sum(teacher_layer_weight)
        for i in range(tea_layer_num):
            if teacher_layer_weight[i] != 0:
                teacher_layer_weight[i] = weight_sum / teacher_layer_weight[i]
                
        student_layer_weight = student_layer_weight / np.sum(student_layer_weight)
        teacher_layer_weight = teacher_layer_weight / np.sum(teacher_layer_weight)

        if type_update == 'att':
            self.att_student_weight = student_layer_weight
            self.att_teacher_weight = teacher_layer_weight
        else:
            self.rep_student_weight = student_layer_weight
            self.rep_teacher_weight = teacher_layer_weight
        
    def embedding_rep_loss(self, student_reps, teacher_reps, 
                           student_layer_weight, teacher_layer_weight,
                           stu_layer_num, tea_layer_num, loss_mse, ):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_rep = student_reps[i]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j]
                tmp_loss = loss_mse(student_rep, teacher_rep)
                # tmp_loss = torch.nn.functional.normalize(tmp_loss, p=2, dim=2)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss
                
        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        # trans_matrix = trans_matrix
        rep_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return rep_loss, trans_matrix, distance_matrix
    
    def emd_rep_loss(self, student_reps, teacher_reps, 
                     student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_rep = student_reps[i+1]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j + 1]
                tmp_loss = loss_mse(student_rep, teacher_rep)
                # tmp_loss = torch.nn.functional.normalize(tmp_loss, p=2, dim=2)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        # trans_matrix = trans_matrix
        rep_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return rep_loss, trans_matrix, distance_matrix

    def emd_att_loss(self, student_atts, teacher_atts, 
                     student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):

        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_att = student_atts[i]
            for j in range(tea_layer_num):
                teacher_att = teacher_atts[j]
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                          student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                          teacher_att)

                tmp_loss = loss_mse(student_att, teacher_att)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss
        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        att_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return att_loss, trans_matrix, distance_matrix
    
    def forward(self, student_atts, teacher_atts, student_reps, teacher_reps,
                device, loss_mse, args, T=1):
        stu_layer_num = len(student_atts)
        tea_layer_num = len(teacher_atts)
        
        if args.use_att:
            att_loss, att_trans_matrix, att_distance_matrix = \
                self.emd_att_loss(student_atts, teacher_atts, self.att_student_weight, self.att_teacher_weight,
                                  stu_layer_num, tea_layer_num, device, loss_mse)
            
            if args.update_weight:
                self.get_new_layer_weight(att_trans_matrix, att_distance_matrix, stu_layer_num, tea_layer_num, T=T)
            att_loss = att_loss.to(device)
        else:
            att_loss = torch.tensor(0)
            
        if args.use_rep:
            if args.embedding_emd:
                rep_loss, rep_trans_matrix, rep_distance_matrix = \
                    self.embedding_rep_loss(student_reps, teacher_reps, self.rep_student_weight, self.rep_teacher_weight,
                                            stu_layer_num+1, tea_layer_num+1, loss_mse)
                if args.update_weight:
                    self.get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num+1, tea_layer_num+1, T=T, type_update='att')
            else:
                rep_loss, rep_trans_matrix, rep_distance_matrix = \
                    self.emd_rep_loss(student_reps, teacher_reps, self.rep_student_weight, self.rep_teacher_weight,
                                      stu_layer_num, tea_layer_num, device, loss_mse)

                if args.update_weight:
                    self.get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num, T=T, type_update='rep')
            rep_loss = rep_loss.to(device)
        else:
            rep_loss = torch.tensor(0)
            
        if not args.seperate:
            student_weight = np.mean(np.stack([self.att_student_weight, self.rep_student_weight]), 0)
            teacher_weight = np.mean(np.stack([self.att_teacher_weight, self.rep_teacher_weight]), 0)
            self.att_student_weight = student_weight
            self.att_teacher_weight = teacher_weight
            self.rep_student_weight = student_weight
            self.rep_teacher_weight = teacher_weight
            
        return att_loss, rep_loss