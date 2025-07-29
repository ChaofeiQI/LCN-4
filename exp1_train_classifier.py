import argparse, os, yaml, warnings, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sync_batchnorm import convert_model
warnings.filterwarnings("ignore")
import datasets, models, utils
import utils.few_shot as fs
from datasets.z_samplers import CategoriesSampler
sys.path.append('./Synchronized-BatchNorm')

def main(config):
    svname = config['name']
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier': svname += '-' + clsfr
    if args.tag is not None: svname += '_' + args.tag
    save_path = os.path.join('./checkpoints', svname)
    utils.ensure_path(save_path,resume = config.get('resume'))
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))
    utils.log('class_total_loss_coeff: {}  meta_loss_coeff: {}'.format(config.get('class_total_loss_coeff'), config.get('meta_loss_coeff')))

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']
    if config.get('n_train_way') is not None: n_train_way = config['n_train_way']
    else: n_train_way = n_way
    if config.get('n_train_shot') is not None: n_train_shot = config['n_train_shot']
    else: n_train_shot = n_shot
    if config.get('ep_per_batch') is not None: ep_per_batch = config['ep_per_batch']
    else: ep_per_batch = 1
    eval_val = eval_fs = False
    if config.get('eval_val') == True: eval_val = True 
    if config.get('eval_fs') == True:  eval_fs = True 

    if config.get('train_dataset'):
        train_dataset = datasets.make(config['train_dataset'], **config['train_dataset_args'])
        train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=args.num_workers)
        utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes))
        if config.get('visualize_datasets'): utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    if config.get('val_dataset'):
        val_dataset = datasets.make(config['val_dataset'], **config['val_dataset_args'])
        utils.log('val dataset (actually few-shot): {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset), val_dataset.n_classes))
        fs_sampler = CategoriesSampler(val_dataset.label, 200, n_way, n_shot + 15, ep_per_batch=4)
        val_loader = DataLoader(val_dataset, batch_sampler=fs_sampler)
    if config.get('fs_dataset_train'):
        fs_dataset_train = datasets.make(config['fs_dataset_train'], **config['fs_dataset_train_args'])
        utils.log('fs train dataset: {} (x{}), {}'.format(fs_dataset_train[0][0].shape, len(fs_dataset_train), fs_dataset_train.n_classes))
        fs_sampler = CategoriesSampler(fs_dataset_train.label, config['train_batches'], n_train_way, n_train_shot + n_query, ep_per_batch=ep_per_batch)
        fs_train_loader = DataLoader(fs_dataset_train, batch_sampler=fs_sampler,num_workers=args.num_workers)
    if config.get('fs_dataset_val'):
        fs_dataset_val = datasets.make(config['fs_dataset_val'], **config['fs_dataset_val_args'])
        utils.log('fs val dataset: {} (x{}), {}'.format(fs_dataset_val[0][0].shape, len(fs_dataset_val),fs_dataset_val.n_classes))
        fs_sampler = CategoriesSampler(fs_dataset_val.label, 200, n_way, n_shot + 15, ep_per_batch=4)
        fs_val_loader = DataLoader(fs_dataset_val, batch_sampler=fs_sampler)

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else: model = models.make(config['model'], **config['model_args'])
    if config.get('_parallel'):
        model = nn.DataParallel(model)
        model = convert_model(model).to('cuda')
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    opt = config['opt']
    if opt == 'adam':
        optimizer, lr_scheduler = utils.make_optimizer( model.parameters(), 'adam', lr= 0.001, weight_decay= 5.e-4)
    elif opt == 'lee': 
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        lambda_epoch = lambda e: 1.0 if e < 40 else (0.06 if e < 60 else 0.012 if e < 80 else (0.0024))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    elif opt == 'lee-modified':
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 24 else 0.012 if e < 26 else (0.0024))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    else: raise ValueError()
    print(model)
        
    for name, param in model.named_parameters():
        if param.requires_grad: print(f"Layer {name}: {param.numel()} parameters")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()
    if config.get('resume'):
        ckpt = torch.load(config['resume'])
        if config.get('_parallel'): model.load_state_dict(utils.sd_parallelize(ckpt['model_sd']))
        else: model.load_state_dict(ckpt['model_sd'])
        optimizer.load_state_dict(ckpt['training']['optimizer_sd'])
        lr_scheduler.load_state_dict(ckpt['training']['lr_scheduler_sd'])
        cur_epoch = ckpt['training']['epoch'] + 1
    else: 
        cur_epoch = 1
        print("No checkpoint loaded, training from scratch.")

    for epoch in range(cur_epoch, max_epoch + 1 + 1):
        np.random.seed(epoch)
        if epoch == max_epoch + 1:
            if not config.get('epoch_ex'): break
            train_dataset.transform = train_dataset.default_transform
            train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True)
        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va', 'f-tl', 'f-ta', 'f-vl', 'f-va']
        aves = {k: utils.Averager() for k in aves_keys}

        ##########################################################################
        # 2.1 Train models
        ########################################################################## 
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        if config.get('train_dataset'): class_iter = iter(train_loader)      # 非小样本阶段
        if config.get('fs_dataset_train'): meta_iter = iter(fs_train_loader) # 小样本阶段
        added_aves_keys = False
        
        for iteration in range(config['train_batches']):
            if config.get('train_dataset'):
                try: class_data, class_label = next(class_iter)
                except StopIteration:
                    class_iter = iter(train_loader)
                    class_data, class_label = next(class_iter)
            if config.get('fs_dataset_train'):
                try: meta_data, _ = next(meta_iter)
                except StopIteration:
                    meta_iter = iter(fs_train_loader)
                    meta_data, _ = next(meta_iter)

            loss_list = []
            if config.get('train_branch_1') and config.get('train_dataset'):
                class_data, class_label = class_data.cuda(), class_label.cuda()
                class_logits, class_sideout_logits = model(mode='class', x=class_data)
                class_loss = F.cross_entropy(class_logits, class_label)
                class_sideout_losses = {k:F.cross_entropy(v, class_label) for k, v in class_sideout_logits.items()}
                
                if args.disable_loss: class_loss = class_loss.detach()
                class_total_loss = class_loss + sum(class_sideout_losses.values())
                class_acc = utils.compute_acc(class_logits, class_label)                
                
                if config.get('class_total_loss_coeff'): 
                    class_total_loss_coeff = config['class_total_loss_coeff']
                else: class_total_loss_coeff = 1.0
                
                loss_list.append(class_total_loss_coeff * class_total_loss)                

                aves['tl'].add(class_total_loss.item())
                aves['ta'].add(class_acc)
                
                if not added_aves_keys:
                    added_aves_keys = True
                    aves['tl-loss'] = utils.Averager()
                    for k in class_sideout_losses: aves['tl-{}'.format(k)] = utils.Averager()
                
                aves['tl-loss'].add(class_loss.item())
                for k in class_sideout_losses: aves['tl-{}'.format(k)].add(class_sideout_losses[k].item())

            if config.get('train_branch_2')  and config.get('fs_dataset_train'):
                x_shot, x_query = fs.split_shot_query(meta_data.cuda(), n_train_way, n_train_shot, n_query, ep_per_batch=ep_per_batch)
                meta_label = fs.make_nk_label(n_train_way, n_query, ep_per_batch=ep_per_batch).cuda()
                meta_logits = model(mode='meta', x_shot=x_shot, x_query=x_query).view(-1, n_train_way)
                meta_loss = F.cross_entropy(meta_logits, meta_label)
                meta_acc = utils.compute_acc(meta_logits, meta_label)
                
                if config.get('meta_loss_coeff'): meta_loss_coeff = config['meta_loss_coeff']
                else: meta_loss_coeff = 1.0
                loss_list.append(meta_loss_coeff * meta_loss)
                aves['f-tl'].add(meta_loss.item()) 
                aves['f-ta'].add(meta_acc)

            optimizer.zero_grad()
            torch.autograd.backward(loss_list)
            optimizer.step()

            class_logits = None; class_sideout_logits = None; class_loss = None; class_sideout_losses = None; class_total_loss = None
            meta_logits = None; meta_loss = None

            print('\r','iteration {} / {}'.format(iteration, config['train_batches']), end="", flush=True)


        if eval_val:
            model.eval()
            for data, label in tqdm(val_loader, desc='val', leave=False):
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    logits, sideout_logits = model(mode='class', x=data, branch=1)
                    loss = F.cross_entropy(logits, label)
                    sideout_losses = {k:F.cross_entropy(v, label) for k, v in sideout_logits.items()}
                    total_loss = loss + sum(sideout_losses.values())
                    acc = utils.compute_acc(logits, label)
                aves['vl'].add(total_loss.item())
                aves['va'].add(acc)
        if eval_fs:
            model.eval()
            np.random.seed(0)
            for data, _ in tqdm(fs_val_loader, desc='f-val' + str(n_shot), leave=False):
                x_shot, x_query = fs.split_shot_query(data.cuda(), n_way, n_shot, 15, ep_per_batch=4)
                label = fs.make_nk_label(n_way, 15, ep_per_batch=4).cuda()
                with torch.no_grad():
                    logits = model(mode='meta', x_shot=x_shot, x_query=x_query, branch=2).view(-1, n_way)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                aves['f-vl'].add(loss.item())
                aves['f-va'].add(acc)

        if lr_scheduler is not None: lr_scheduler.step()
        for k, v in aves.items(): aves[k] = v.item()
        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        if epoch <= max_epoch: epoch_str = str(epoch)
        else: epoch_str = 'ex'

        if config.get('train_branch_1') and not config.get('train_branch_2'): 
            log_str = 'epoch {}/{}, train-class {:.4f}|{:.4f}'.format(epoch_str, max_epoch, aves['tl'], aves['ta'])
        elif config.get('train_branch_2') and not config.get('train_branch_1'): 
            log_str = 'epoch {}/{}, train-meta {:.4f}|{:.4f}'.format(epoch_str, max_epoch, aves['f-tl'], aves['f-ta'])
        elif config.get('train_branch_1') and config.get('train_branch_2'): 
            log_str = 'epoch {}/{}, train-class {:.4f}|{:.4f},  train-meta {:.4f}|{:.4f}'.format(epoch_str, max_epoch, aves['tl'], aves['ta'], aves['f-tl'], aves['f-ta'])
        else: raise ValueError()
        log_str += '(' + ' '.join('{}:{}'.format(k, aves[k]) for k in aves if k.startswith('tl-')) + ')'
        if eval_val: log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
        if eval_fs: log_str += ', fs {:.4f}|{:.4f}'.format(aves['f-vl'], aves['f-va'])
        if epoch <= max_epoch: log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else: log_str += ', {}'.format(t_epoch)        
        utils.log(log_str)

        if config.get('_parallel'): model_ = model.module
        else: model_ = model
        training = {
            'epoch': epoch,
            'optimizer': config['opt'],
            'optimizer_args': config['opt'],
            'optimizer_sd': optimizer.state_dict(),
            'lr_scheduler_sd': lr_scheduler.state_dict()}
        save_obj = {
            'file': __file__,
            'config': config,
            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),
            'training': training,}
        if epoch <= max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
            if (save_epoch is not None) and epoch % save_epoch == 0: torch.save(save_obj, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
            if aves['f-va'] > max_va:
                max_va = aves['f-va']
                torch.save(save_obj, os.path.join(save_path, 'max-f-va.pth'))
        else: torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_classifier_mini.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--vscode_debug', action='store_true', default=False)
    parser.add_argument('--max_epoch', default=-1, type=int)
    parser.add_argument('--disable_loss', action='store_true', default=False)
    parser.add_argument('--train_batches', default=-1, type=int)
    parser.add_argument('--resume',default=None)
    args = parser.parse_args()

    if args.vscode_debug:
        import ptvsd
        print("Enabling attach starts.")
        ptvsd.enable_attach(address=('0.0.0.0', 9310))
        ptvsd.wait_for_attach()
        print("Enabling attach ends.")

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    if args.max_epoch != -1: config['max_epoch'] = args.max_epoch
    if args.train_batches > 0: config['train_batches'] = args.train_batches
    config['resume'] = args.resume    
    utils.set_gpu(args.gpu)

    main(config)