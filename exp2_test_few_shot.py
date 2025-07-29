import argparse, os, yaml, warnings, sys, scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sync_batchnorm import convert_model
from pathlib import Path
from os.path import join
warnings.filterwarnings("ignore")
import datasets, models, utils
import utils.few_shot as fs
from datasets.z_samplers import CategoriesSampler
sys.path.append('./Synchronized-BatchNorm')
from scipy.spatial.distance import braycurtis
from sklearn.metrics import confusion_matrix

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

# BCD: Differentiable Bray Curtis Distance
def metametric(qry, sup):
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
    dbcd_distance = dbcd_distance[:, :, :sup_fea.size(1)]
    return dbcd_distance *(-1)

def main(config):
    utils.log('test shot: {}'.format(args.shot))
    utils.log('dist method: {} logits_coeff_list: {} , feat_source_list: {} branch_list: {}'.format(args.method, args.logits_coeff_list, args.feat_source_list, args.branch_list))

    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(dataset[0][0].shape, len(dataset), dataset.n_classes))
    n_way = 5
    n_shot, n_query = args.shot, 15
    n_batch = 800    
    ep_per_batch = 1 
    batch_sampler = CategoriesSampler(dataset.label, n_batch, n_way, n_shot + n_query, ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

    models_list = []
    if config.get('load') is not None:
        model_data = torch.load(config.get('load'))
        model = models.load(model_data)
        if config.get('_parallel'):
            model = nn.DataParallel(model)
            model = convert_model(model).to('cuda')
    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}
    test_epochs = args.test_epochs
    np.random.seed(0)
    va_lst, cm_list = [],[]
    
    for epoch in range(1, test_epochs + 1):
        for data, _ in tqdm(loader, leave=False):
            x_shot_origin, x_query_origin = fs.split_shot_query(data.cuda(), n_way, n_shot, n_query, ep_per_batch=ep_per_batch)
            with torch.no_grad():           
                logits_list = []  
                method_i_list = args.method.split(',')   
                logits_coeff_list = list(map(eval, args.logits_coeff_list.split(',')))
                branch_list = list(map(int, args.branch_list.split(',')))
                feat_source_list = args.feat_source_list.split(',')
                method_logits_list = []
                for method_i, logits_coeff, feat_source, branch in zip(method_i_list, logits_coeff_list, feat_source_list, branch_list):
                    x_query_out, x_shot_out, s_query_out, s_shot_out = model(mode='meta_test',
                                                x_shot=x_shot_origin, x_query=x_query_origin, sideout=True, branch=branch)
                    if feat_source == 'final':
                        x_query_pre = x_query_out
                        x_shot_pre  = x_shot_out
                        x_shot_pre = x_shot_pre.mean(dim=2)                                           
                        x_shot = x_shot_pre.view(*x_shot_pre.shape[:2], -1)    
                        x_query = x_query_pre.view(*x_query_pre.shape[:2], -1) 
                        assert x_query.dim() == x_shot.dim() == 3
                        if method_i == 'cos':
                            x_shot_processed = F.normalize(x_shot, dim=-1)     
                            x_query_processed = F.normalize(x_query, dim=-1)   
                            logits = torch.bmm(x_query_processed, x_shot_processed.permute(0, 2, 1))
                        elif method_i == 'bcd':
                            x_shot_processed = F.normalize(x_shot, dim=-1)      
                            x_query_processed = F.normalize(x_query, dim=-1)    
                            logits =  metametric(x_query_processed, x_shot_processed)
                        else: raise NotImplementedError()

                    else:
                        x_query_pre = s_query_out[feat_source]  
                        x_shot_pre = s_shot_out[feat_source]
                        x_shot_pre = x_shot_pre.mean(dim=2)  
                        x_shot = x_shot_pre.view(*x_shot_pre.shape[:2], -1)       
                        x_query = x_query_pre.view(*x_query_pre.shape[:2], -1)    
                        assert x_query.dim() == x_shot.dim() == 3
                        if method_i == 'cos':
                            x_shot_processed = F.normalize(x_shot, dim=-1)         
                            x_query_processed = F.normalize(x_query, dim=-1)      
                            logits = torch.bmm(x_query_processed, x_shot_processed.permute(0, 2, 1))
                        elif method_i == 'bcd':
                            x_shot_processed = F.normalize(x_shot, dim=-1) 
                            x_query_processed = F.normalize(x_query, dim=-1)
                            logits =  metametric(x_query_processed, x_shot_processed)                            
                        else: raise NotImplementedError()
                    method_logits_list.append(logits * logits_coeff)

                logits = sum(method_logits_list).view(-1, n_way)
                label = fs.make_nk_label(n_way, n_query, ep_per_batch=ep_per_batch).cuda()
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)
                aves['vl'].add(loss.item(), len(data))
                aves['va'].add(acc, len(data))
                va_lst.append(acc)

                _, y_pred = torch.max(logits.cpu(), dim=1)
                cm = confusion_matrix(label.cpu(), y_pred)
                cm_list.append(cm/15)#Add confusion matrix

        if epoch==10:
            sum_cm = np.zeros_like(cm_list[0], dtype=float)
            for cm in cm_list: sum_cm += cm
            avg_cm = sum_cm / len(cm_list)
            # print("Confusion Matrix= \n", avg_cm)
            # print("Confusion Matrix{:.4f}= \n", avg_cm)

        utils.log('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f}'.format(epoch, aves['va'].item() * 100, mean_confidence_interval(va_lst)*100,  aves['vl'].item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-epochs', type=int, default=10)   
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--method', default='cos') 
    parser.add_argument('--load_encoder', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--log_filename', type=str, default='log.txt')
    parser.add_argument('--vscode_debug', action='store_true', default=False)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--logits_coeff_list', type=str, default='1.0')
    parser.add_argument('--sideout', action='store_true', default=False)
    parser.add_argument('--feat_source_list', type=str, default='')
    parser.add_argument('--branch_list', type=str, default='')
    args = parser.parse_args()

    if args.vscode_debug:
        import debugpy
        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    config = yaml.load(open(args.config,'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1: config['_parallel'] = True
    if args.load_encoder: config['load_encoder'] = args.load_encoder
    if args.load: config['load'] = args.load
    if args.save_path: 
        os.makedirs(args.save_path, exist_ok=True)
        utils.set_log_path(args.save_path)
    else:
        load_path = Path(args.load)        
        model_dir = str(load_path.parent)
        utils.set_log_path(model_dir)
    utils.set_log_filename(args.log_filename)
    utils.set_gpu(args.gpu)

    main(config)