import os

import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from datasets import EAVDataset, AFFECDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, Model, MaskedMSELoss, FocalLoss
from softcomputing import FuzzyRoughSelector, rough_sample_weights, weighted_nll
import json
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import pandas as pd
import pickle as pk
import datetime

print("import OK!")

seed = 1475 
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    np.random.seed(int(seed)+worker_id)

def get_train_valid_sampler(trainset, valid=0, dataset='EAV'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def stratified_split(labels, valid_frac, seed=seed, groups=None):
    """Random class-stratified train/validation split (validation never touches the test set).
    If `groups` is given (e.g. trial ids of time windows) whole groups are kept together."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels).astype(int).ravel()
    if groups is not None:
        ug, first = np.unique(groups, return_index=True)
        _, gva = stratified_split(labels[first], valid_frac, seed)
        val = np.where(np.isin(groups, ug[gva]))[0]
        return np.setdiff1d(np.arange(len(labels)), val), val
    val = []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        val.extend(idx[:int(round(valid_frac*len(idx)))])
    val = np.array(sorted(val), dtype=int)
    tr = np.setdiff1d(np.arange(len(labels)), val)
    return tr, val

def _make_loaders(DS, batch_size, valid, num_workers, pin_memory, sub):
    trainset = DS(train=True, subject=sub)
    testset = DS(train=False, subject=sub)
    if valid > 0:
        tr_idx, va_idx = stratified_split(trainset.labels(True), valid, groups=trainset.groups(True))
        train_sampler, valid_sampler = SubsetRandomSampler(tr_idx), SubsetRandomSampler(va_idx)
    else:  # original behaviour: train on everything, empty validation loader
        tr_idx, va_idx = np.arange(len(trainset)), np.array([], dtype=int)
        train_sampler, valid_sampler = None, SubsetRandomSampler([])
    trainset.fit_idx = tr_idx  # samples that soft-computing steps are allowed to fit on
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, collate_fn=trainset.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=_init_fn)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, collate_fn=trainset.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=_init_fn)
    return train_loader, valid_loader, test_loader

def get_EAV_loaders(batch_size=16, valid=0, num_workers=0, pin_memory=False, sub=1):
    return _make_loaders(EAVDataset, batch_size, valid, num_workers, pin_memory, sub)

def get_AFFEC_loaders(batch_size=16, valid=0, num_workers=0, pin_memory=False, sub='acl'):
    return _make_loaders(AFFECDataset, batch_size, valid, num_workers, pin_memory, sub)


def apply_rough_selection(trainset, testset, method, sigma, max_features, pool):
    """Fuzzy-rough reduct per modality, fitted on the training split only, applied to train+test.
    Returns (selectors, per-sample loss weights from the positive-region membership)."""
    fit = trainset.fit_idx
    y = trainset.labels(True)[fit]
    selectors, mus = {}, []
    for mod in ('eeg', 'aud', 'vis'):
        X = trainset.mod_array(mod, True)[fit]
        sel = FuzzyRoughSelector(sigma=sigma, pool_size=pool, max_features=max_features).fit(X, y, method=method, verbose=False)
        print('[rough-set] %s: %d -> %d features (gamma=%.3f)' % (mod, X.shape[1], len(sel.selected_), sel.gamma_))
        mus.append(sel.positive_region_membership(X, y))
        trainset.select_features(mod, sel.selected_)
        testset.select_features(mod, sel.selected_)
        selectors[mod] = sel
    w = np.ones(len(trainset), dtype=np.float32)
    w[fit] = rough_sample_weights(np.mean(mus, axis=0), floor=args.rs_floor)
    return selectors, w

GATE_LOG = {'w': [], 'y': []}  # fuzzy-gate weights collected during evaluation

def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='EAV', sample_weights=None):
    losses, preds, labels = [], [], []  
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), [] 

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda() 

    assert not train or optimizer!=None 
    if train:
        model.train() 
    else:
        model.eval()

    if not train:
        GATE_LOG['w'], GATE_LOG['y'] = [], []
    seed_everything() 
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        eegf1,eegf2,eegf3,eegf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        if args.multi_modal:  # 如果启用了多模态
            if args.mm_fusion_mthd=='concat':  # 如果融合方法是连接
                if modals == 'avl':
                    eegf = torch.cat([acouf, visuf, eeg1, eeg2, eeg3, eeg4],dim=-1)  
                elif modals == 'av':
                    eegf = torch.cat([acouf, visuf],dim=-1)    # 连接音频和视觉特征
                elif modals == 'vl':
                    eegf = torch.cat([visuf, eegf1,eegf2,eegf3,eegf4],dim=-1)   # 连接视觉和文本特征
                elif modals == 'al':
                    eegf = torch.cat([acouf, eegf1,eegf2,eegf3,eegf4],dim=-1)  # 连接音频和文本特征
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd=='gated':
                eegf = eegf   
        else:  
            if modals == 'a':  
                eegf = acouf
            elif modals == 'v': 
                eegf = visuf
            elif modals == 'l': 
                eegf = eegf
            else:
                raise NotImplementedError

        # 计算每个序列的有效长度
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

        # 调整形状
        eegf1 = eegf1.permute(1, 0).unsqueeze(0)  # [1, 16, 1024]
        eegf2 = eegf2.permute(1, 0).unsqueeze(0)  # [1, 16, 1024]
        eegf3 = eegf3.permute(1, 0).unsqueeze(0)  # [1, 16, 1024]
        eegf4 = eegf4.permute(1, 0).unsqueeze(0)  # [1, 16, 1024]
        qmask = qmask.permute(1, 0).unsqueeze(0)  # [1, 16, 2]
        acouf = acouf.permute(1, 0).unsqueeze(0)  # [1, 16, 1582]
        visuf = visuf.permute(1, 0).unsqueeze(0)  # [1, 16, 342]



        # 前向传播
        if args.multi_modal and args.mm_fusion_mthd=='gated':
            log_prob, e_i, e_n, e_t, e_l = model(eegf, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_subsequently':   
            log_prob, e_i, e_n, e_t, e_l = model([eegf1,eegf2,eegf3,eegf4], qmask, umask, lengths, acouf, visuf, epoch)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_DHT':   
            log_prob, e_i, e_n, e_t, e_l = model([eegf1,eegf2,eegf3,eegf4], qmask, umask, lengths, acouf, visuf, epoch)
        else:
            log_prob, e_i, e_n, e_t, e_l = model(eegf, qmask, umask, lengths)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))]) 
        label = label.long()

        sw = torch.as_tensor(sample_weights[np.array(data[-1])]) if (train and sample_weights is not None) else None
        cw = getattr(loss_function, 'weight', None)
        if sw is not None:  # rough-set boundary samples contribute less
            loss = weighted_nll(log_prob, label, cw, sw)
        else:
            loss = loss_function(log_prob, label)
        if getattr(model, 'fuzzy_gate', None) is not None:
            if train:  # deep supervision of the modality-specific heads that feed the fuzzy gate
                loss = loss + args.aux_weight * model.fuzzy_gate.aux_loss(label, lambda lp, l: weighted_nll(lp, l, cw, sw))
            else:
                GATE_LOG['w'].append(model.fuzzy_gate.last_weights.cpu().numpy()); GATE_LOG['y'].append(label.cpu().numpy())
        preds.append(torch.argmax(log_prob, 1).cpu().numpy()) 
        labels.append(label.cpu().numpy()) 
        losses.append(loss.item()) 
        if train:
            loss.backward()  
            optimizer.step()  
            
    if preds!=[]:
        preds  = np.concatenate(preds) 
        labels = np.concatenate(labels) 
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], [] # 如果没有预测结果，返回NaN

    vids += data[-1]   
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4) 
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2) 
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2) 

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el  


if __name__ == '__main__':
    path = './saved/EAV/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU') 

    parser.add_argument('--base-model', default='GRU', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')  

    parser.add_argument('--graph-model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True, help='whether to use nodal attention in graph model') 

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances') 

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances') 

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')  
    
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')  
    
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model') 
    
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='hyper', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')  

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information') 

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')  

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads') 

    parser.add_argument('--graph_construct', default='direct', help='single/window/fc for MMGCN2; direct/full for others') 

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')  

    parser.add_argument('--use_residue', action='store_true', default=False, help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=True, help='whether to use multimodal information') 

    parser.add_argument('--mm_fusion_mthd', default='concat_DHT', help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')  

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')  

    parser.add_argument('--Dataset', default='EAV', help='dataset to train and test, EAV/AFFEC') 

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding') 

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding') 

    parser.add_argument('--norm', default='BN', help='NORM type')  

    parser.add_argument('--testing', action='store_true', default=False, help='testing') 

    parser.add_argument('--num_L', type=int, default=3, help='num_hyperconvs') 

    parser.add_argument('--num_K', type=int, default=0, help='num_convs')

    parser.add_argument('--subject', type=int, default=1, help='subject id')  

    # ---- soft-computing options (all off by default => original Hyper-MML behaviour) ----
    parser.add_argument('--rs_select', default='none', choices=['none', 'quickreduct', 'ga'], help='fuzzy-rough feature reduction (QuickReduct or GA search)')
    parser.add_argument('--rs_sigma', type=float, default=3.0, help='width of fuzzy similarity relation (in std)')
    parser.add_argument('--rs_max_features', type=int, default=48, help='max reduct size per modality')
    parser.add_argument('--rs_pool', type=int, default=200, help='candidate pool size after ANOVA pre-filter')
    parser.add_argument('--rs_weight', action='store_true', default=False, help='down-weight rough boundary-region samples in the loss')
    parser.add_argument('--rs_floor', type=float, default=0.3, help='minimum loss weight of boundary samples')
    parser.add_argument('--fuzzy_gate', action='store_true', default=False, help='fuzzy TSK modality-reliability gate before hypergraph fusion')
    parser.add_argument('--aux_weight', type=float, default=0.3, help='weight of auxiliary modality-head loss for the fuzzy gate')
    parser.add_argument('--seed', type=int, default=1475, help='random seed (model init / sampling)')
    parser.add_argument('--valid_frac', type=float, default=0.0, help='fraction of train set held out for validation (needed for honest model selection / tuning)')
    args = parser.parse_args() 
    seed_everything.__defaults__ = (args.seed,)  # seed_everything() is called without arguments throughout
    today = datetime.datetime.now() 
    subject = args.subject
    print(args) 

    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+'using_lstm_'+args.Dataset
    else:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+str(args.Deep_GCN_nlayers)+'_'+args.Dataset

    if args.use_speaker:
        name_ = name_+'_speaker'
    if args.use_modal:
        name_ = name_+'_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda  
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter 
        writer = SummaryWriter()

    cuda       = args.cuda 
    n_epochs   = args.epochs 
    batch_size = args.batch_size  
    modals = args.modals  

    # ---- data loaders (built before the model so that feature selection can change input dims) ----
    print(f'-------------subject{subject}------------')
    if args.Dataset == 'EAV':
        train_loader, valid_loader, test_loader = get_EAV_loaders(valid=args.valid_frac, batch_size=batch_size,
                                                                  num_workers=0, sub=subject)
    elif args.Dataset == 'AFFEC':
        train_loader, valid_loader, test_loader = get_AFFEC_loaders(valid=args.valid_frac, batch_size=batch_size,
                                                                    num_workers=0, sub=subject)
    else:
        raise ValueError("There is no such dataset")

    # input dimensions are read from the data (EAV: 1582/1024/1024; AFFEC pretrained features: 31/44/1024)
    _ds = train_loader.dataset
    D_audio, D_visual, D_eeg = (_ds.mod_array(m, True).shape[1] for m in ('aud', 'vis', 'eeg'))
    train_sample_w = None
    if args.rs_select != 'none':  # rough-set / fuzzy-rough feature reduction
        selectors, w_all = apply_rough_selection(train_loader.dataset, test_loader.dataset, args.rs_select,
                                                 args.rs_sigma, args.rs_max_features, args.rs_pool)
        D_eeg, D_audio, D_visual = len(selectors['eeg'].selected_), len(selectors['aud'].selected_), len(selectors['vis'].selected_)
        if args.rs_weight:
            train_sample_w = w_all
    elif args.rs_weight:
        raise SystemExit('--rs_weight needs --rs_select quickreduct|ga')

    if args.multi_modal:   
        if args.mm_fusion_mthd=='concat':
            if modals == 'avl':
                D_m = D_audio+D_visual+D_eeg
            elif modals == 'av':
                D_m = D_audio+D_visual
            elif modals == 'al':
                D_m = D_audio+D_eeg
            elif modals == 'vl':
                D_m = D_visual+D_eeg
            else:
                raise NotImplementedError
        else:
            D_m = D_eeg
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_eeg
        else:
            raise NotImplementedError

    D_g = 512  
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    n_speakers = 2
    n_classes  = 5 if args.Dataset=='EAV' else 3

    if args.graph_model:  
        seed_everything()

        model = Model(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K,
                                 fuzzy_gate = args.fuzzy_gate)

        print ('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic LSTM Model.')

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()

    if args.Dataset == 'EAV':
        loss_weights = torch.FloatTensor([1/0.2,
                                        1/0.2,
                                        1/0.2,
                                        1/0.2,
                                        1/0.2])
    if args.Dataset == 'AFFEC':
        loss_weights = torch.FloatTensor([1/0.33,
                                        1/0.34,
                                        1/0.33])


    if args.class_weight: 
        if args.graph_model:
            loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights) # 使用负对数似然损失
        else:
            loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        if args.graph_model:
            loss_function = nn.NLLLoss()
        else:
            loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)  # 初始化优化器

    lr = args.lr

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None  
    all_fscore, all_acc, all_loss = [], [], [] 
    best_valid, test_at_best_valid = None, (float('nan'), float('nan'), -1)

    if args.testing: 
        state = torch.load("best_model.pth.tar")
        model.load_state_dict(state)
        print('testing loaded model')
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, 0, cuda, args.modals, dataset=args.Dataset)
        print('test_acc:',test_acc,'test_fscore:',test_fscore)


    for e in range(n_epochs):
        start_time = time.time()  

        if args.graph_model: 
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, train_loader, e, cuda, args.modals, optimizer, True, dataset=args.Dataset, sample_weights=train_sample_w)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, valid_loader, e, cuda, args.modals, dataset=args.Dataset)
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
            all_fscore.append(test_fscore) 
            all_acc.append(test_acc)
            if args.valid_frac > 0 and (best_valid is None or valid_fscore > best_valid):  # honest model selection
                best_valid, test_at_best_valid = valid_fscore, (test_fscore, test_acc, e + 1)


        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)
            all_fscore.append(test_fscore)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred = test_loss, test_label, test_pred

        if best_fscore == None or best_fscore < test_fscore: 
            best_fscore = test_fscore
            best_label, best_pred = test_label, test_pred
            #test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))


    if args.tensorboard:
        writer.close()

    if args.fuzzy_gate and GATE_LOG['w']:
        gw, gy = np.concatenate(GATE_LOG['w']), np.concatenate(GATE_LOG['y'])
        print('Fuzzy gate - mean modality trust on test set (EEG / AUDIO / VIDEO):')
        print('  overall:', np.round(gw.mean(0), 3))
        for c in np.unique(gy):
            print('  class %d:' % c, np.round(gw[gy == c].mean(0), 3))
        print('Learned fuzzy sets and rule base:')
        print(model.fuzzy_gate.describe())
    result = {'test_f1_max_over_epochs': max(all_fscore), 'test_f1_last_epoch': all_fscore[-1],
              'test_acc_last_epoch': all_acc[-1], 'best_valid_f1': best_valid,
              'test_f1_at_best_valid': test_at_best_valid[0], 'test_acc_at_best_valid': test_at_best_valid[1],
              'best_valid_epoch': test_at_best_valid[2], 'n_features': [D_eeg, D_audio, D_visual]}
    print('RESULT ' + json.dumps(result))

    if not args.testing: 
        print('Test performance..')
        print ('F-Score:', max(all_fscore)) 
        if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
            with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
                pk.dump({}, f)  
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
            record = pk.load(f)  
        key_ = name_
        if record.get(key_, False):
            record[key_].append(max(all_fscore)) 
        else:
            record[key_] = [max(all_fscore)] 
        if record.get(key_+'record', False):
            record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))  
        else:
            record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)] 
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
            pk.dump(record, f)

        print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)) 
        print(confusion_matrix(best_label,best_pred,sample_weight=best_mask)) 
