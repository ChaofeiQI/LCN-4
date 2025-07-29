import math
import torch
import torch.nn as nn
import models
import utils
from .base import register
import torch.nn.functional as F

@register('linear-classifier')
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.linear(x)

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T=4.0):
        super(DistillKL, self).__init__()
        self.T = T
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss

class JS_Divergence_Loss(nn.Module):
    """JS divergence for distillation"""
    def __init__(self, T=4.0):
        super(JS_Divergence_Loss, self).__init__()
        self.kl_divergence = DistillKL(T)
    def forward(self, js_p, js_q):
        m = 0.5 * (js_p + js_q)
        return 0.5 * self.kl_divergence(js_p, m) + 0.5 * self.kl_divergence(js_q, m)


##########################################
#  Locatio-aware Constellation Network
##########################################
# @register('LACN'): Only For Testing on Our Pretrained Models
@register('LCN-4')
class LCN4(nn.Module):
    def __init__(self, encoder, encoder_args, classifier, classifier_args, sideout_info=[], method='sqr', temp=1.0, temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        # Standard classifier.
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)
        self.sideout_info = sideout_info
        self.sideout_classifiers = nn.ModuleList()
        for _, sideout_dim in self.sideout_info:
            classifier_args['in_dim'] = sideout_dim
            self.sideout_classifiers.append(models.make(classifier, **classifier_args))
        # Few-shot classifier.
        self.method = method
        if temp_learnable: self.temp = nn.Parameter(torch.tensor(temp))
        else: self.temp = temp

    def diff_canberra_distance(self, qry, sup):
        qry_fea = qry.view(qry.size(0), qry.size(1), -1)  
        sup_fea = sup.view(sup.size(0), sup.size(1), -1)  
        qry_fea_expand = qry_fea.unsqueeze(2).expand(qry_fea.size(0), qry_fea.size(1), sup_fea.size(1), qry_fea.size(2))
        sup_fea_expand = sup_fea.unsqueeze(1).expand(sup_fea.size(0), qry_fea.size(1), sup_fea.size(1), sup_fea.size(2))
        min_val = torch.min(qry_fea_expand.min(dim=3, keepdim=True)[0], sup_fea_expand.min(dim=3, keepdim=True)[0])
        max_val = torch.max(qry_fea_expand.max(dim=3, keepdim=True)[0], sup_fea_expand.max(dim=3, keepdim=True)[0])
        qry_fea_expand = (qry_fea_expand - min_val) / (max_val - min_val)
        sup_fea_expand = (sup_fea_expand - min_val) / (max_val - min_val)
        numerator = torch.sum(torch.abs(qry_fea_expand - sup_fea_expand), dim=3)
        denominator = torch.sum(qry_fea_expand + sup_fea_expand, dim=3)
        dbcd_distance = numerator/denominator
        dbcd_distance = F.normalize(dbcd_distance, p=2, dim=2)
        similarity = -dbcd_distance 
        similarity = similarity[:, :, :sup_fea.size(1)]
        return similarity

    def diff_bray_curtis_distance(self, qry, sup):
        qry_fea = qry.view(qry.size(0), qry.size(1), -1) 
        sup_fea = sup.view(sup.size(0), sup.size(1), -1) 
        qry_fea_expand = qry_fea.unsqueeze(2).expand(qry_fea.size(0), qry_fea.size(1), sup_fea.size(1), qry_fea.size(2))
        sup_fea_expand = sup_fea.unsqueeze(1).expand(sup_fea.size(0), qry_fea.size(1), sup_fea.size(1), sup_fea.size(2))
        min_val = torch.min(qry_fea_expand.min(dim=3, keepdim=True)[0], sup_fea_expand.min(dim=3, keepdim=True)[0])
        max_val = torch.max(qry_fea_expand.max(dim=3, keepdim=True)[0], sup_fea_expand.max(dim=3, keepdim=True)[0])
        qry_fea_expand = (qry_fea_expand - min_val) / (max_val - min_val)
        sup_fea_expand = (sup_fea_expand - min_val) / (max_val - min_val)
        numerator = torch.abs(qry_fea_expand - sup_fea_expand)
        denominator = torch.abs(qry_fea_expand) + torch.abs(sup_fea_expand) + 1e-15 
        canberra_distance = torch.sum(numerator / denominator, dim=2)
        canberra_distance = F.normalize(canberra_distance, p=2, dim=2)
        similarity = -1* canberra_distance
        similarity = similarity[:, :, :sup_fea.size(1)]
        return similarity

    def metametric(self, qry, sup):
        qry_fea = qry.view(qry.size(0), qry.size(1), -1)
        sup_fea = sup.view(sup.size(0), sup.size(1), -1) 
        qry_fea_expand = qry_fea.unsqueeze(2).expand(qry_fea.size(0), qry_fea.size(1), sup_fea.size(1), qry_fea.size(2))
        sup_fea_expand = sup_fea.unsqueeze(1).expand(sup_fea.size(0), qry_fea.size(1), sup_fea.size(1), sup_fea.size(2))
        min_val = torch.min(qry_fea_expand.min(dim=3, keepdim=True)[0], sup_fea_expand.min(dim=3, keepdim=True)[0])
        max_val = torch.max(qry_fea_expand.max(dim=3, keepdim=True)[0], sup_fea_expand.max(dim=3, keepdim=True)[0])
        qry_fea_expand = (qry_fea_expand - min_val) / (max_val - min_val)
        sup_fea_expand = (sup_fea_expand - min_val) / (max_val - min_val)
        JS_Divergence = JS_Divergence_Loss()(qry_fea_expand, sup_fea_expand)
        numerator = torch.abs(qry_fea_expand - sup_fea_expand)
        denominator = torch.abs(qry_fea_expand) + torch.abs(sup_fea_expand) + 1e-15
        canberra_distance = torch.sum(numerator / denominator, dim=2)
        canberra_distance = canberra_distance[:, :, :sup_fea.size(1)]
        canberra_distance = F.normalize(canberra_distance, p=2, dim=2)
        numerator = torch.sum(torch.abs(qry_fea_expand - sup_fea_expand), dim=3)
        denominator = torch.sum(qry_fea_expand + sup_fea_expand, dim=3)
        dbcd_distance = numerator/denominator
        dbcd_distance = dbcd_distance[:, :, :sup_fea.size(1)]
        return dbcd_distance *(-1)


    def forward(self, mode, x=None, x_shot=None, x_query=None, branch=-1, sideout=False):
        # 1. Non-scenario Training (standard classification).
        def class_forward(x, branch):  
            x, s = self.encoder(x, sideout=True, branch=branch) 
            out_x = self.classifier(x)                         
            out_s = {}
            for i, (sideout_name, _) in enumerate(self.sideout_info):
                feat_s = s[sideout_name]
                if feat_s.dim() == 4:
                    feat_s = feat_s.mean(-1).mean(-1) 
                out_s[sideout_name] = self.sideout_classifiers[i](feat_s)
            return out_x, out_s
        
        # 2. Scenario Training (meta training).
        def meta_forward(x_shot, x_query, branch):
            shot_shape, query_shape = x_shot.shape[:-3], x_query.shape[:-3]
            img_shape = x_shot.shape[-3:]           
            x_shot  = x_shot.view(-1, *img_shape)  
            x_query = x_query.view(-1, *img_shape) 
            x_tot   = self.encoder(torch.cat([x_shot, x_query], dim=0), sideout=False, branch=branch)
            x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]   
            x_shot  = x_shot.view(*shot_shape, -1)                         
            x_query = x_query.view(*query_shape, -1)                
            
            if self.method == 'cos':
                x_shot = x_shot.mean(dim=-2)
                x_shot = F.normalize(x_shot, dim=-1)
                x_query = F.normalize(x_query, dim=-1)
                metric = 'dot'
                logits = utils.compute_logits(x_query, x_shot, metric=metric, temp=self.temp)
            
            elif self.method == 'sqr':
                x_shot = x_shot.mean(dim=-2)
                metric = 'sqr'
                logits = utils.compute_logits(x_query, x_shot, metric=metric, temp=self.temp / 1600.)

            elif self.method == 'bcd':
                x_shot = x_shot.mean(dim=-2)
                x_shot = F.normalize(x_shot, dim=-1)
                x_query = F.normalize(x_query, dim=-1)
                logits = self.metametric(x_query, x_shot)
            return logits

        # Few-shot classifier (for meta test).
        def meta_test_forward(x_shot, x_query, branch, sideout=False):
            shot_shape, query_shape = x_shot.shape[:-3], x_query.shape[:-3]
            img_shape   = x_shot.shape[-3:]           
            x_shot      = x_shot.view(-1, *img_shape) 
            x_query     = x_query.view(-1, *img_shape)
            x_shot_len, x_query_len = len(x_shot), len(x_query)
            
            if sideout: 
                x_tot, s_tot = self.encoder(torch.cat([x_shot, x_query], dim=0), sideout=True, branch=branch)
                x_shot,x_query = x_tot[:x_shot_len], x_tot[-x_query_len:] 
                feat_shape = x_shot.shape[1:] 
                x_shot = x_shot.view(*shot_shape, *feat_shape)    
                x_query = x_query.view(*query_shape, *feat_shape) 
                s_shot  = {k:v[  :x_shot_len].view(*shot_shape,  *v.shape[1:]) for k, v in s_tot.items()}
                s_query = {k:v[-x_query_len:].view(*query_shape, *v.shape[1:]) for k, v in s_tot.items()}
                return x_query, x_shot, s_query, s_shot
            else:
                x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0), sideout=False, branch=branch)
                x_shot, x_query = x_tot[:x_shot_len], x_tot[-x_query_len:]
                feat_shape = x_shot.shape[1:]
                x_shot = x_shot.view(*shot_shape, *feat_shape)
                x_query = x_query.view(*query_shape, *feat_shape)
                return x_query, x_shot

        ###########################################
        # Main function：Training or evaluation
        ###########################################
        if self.training:
            # 1.For standard classification.
            if mode=='class':
                out_x, out_s = class_forward(x, branch=1)
                return out_x, out_s
            # 2.For few-shot classification.
            elif mode=='meta':
                logits = meta_forward(x_shot, x_query, branch=2)
                return logits
            else: raise ValueError()
        else:
            # 1.For standard classification: Validation and test.
            if mode=='class':
                out_x, out_s = class_forward(x, branch=branch)
                return out_x, out_s
            # 2.For few-shot classification: Validation.
            elif mode=='meta':
                logits = meta_forward(x_shot, x_query, branch=branch)
                return logits
            # 3.For few-shot classification: Test(sideout is enabled)
            elif mode=='meta_test': 
                return meta_test_forward(x_shot, x_query, branch=branch, sideout=sideout)
            else: raise ValueError()
