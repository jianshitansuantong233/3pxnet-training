import argparse
import os
import torch
import logging

import numpy as np
import torch.nn as nn

from utils import *
import utils_own
import network

import torch.optim as optim

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')

parser.add_argument('--save_dir', metavar='SAVE_DIR', default='./training_data/mnist/', type=str, help='Save dir')
parser.add_argument('--batch', metavar='BATCH', default=256, type=int, help='Batch size to use')
parser.add_argument('--size', metavar='SIZE', default=0, type=int, help='Size of network. 1 means large and 0 means small, 2 means CNN')
parser.add_argument('--full', metavar='FULL', default=0, type=int, help='If 1, train with full precision')
parser.add_argument('--binary', metavar='BINARY', default=0, type=int, help='If 1, train with dense binary')
parser.add_argument('--first_sparsity', metavar='FIRST_SPARSITY', default=0.99, type=float, help='Sparsity of the first layer')
parser.add_argument('--rest_sparsity', metavar='REST_SPARSITY', default=0.99, type=float, help='Sparsity of other layers')
parser.add_argument('--permute', metavar='PERMUTE', default=-1, type=int, help='Permutation method. -1 means no permutation. 0 uses sort, 1 uses group (method mentioned in paper)')
parser.add_argument('--seed', metavar='SEED', default=0, type=int, help='Seed to use for this run')
parser.add_argument('--cpu', dest='cpu', default=False, action='store_true', help='Train using cpu')
                    
                    

def main():
    global args, best_prec1
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    
    dataset='MNIST'
    first_sparsity = args.first_sparsity
    rest_sparsity = args.rest_sparsity
    permute = args.permute
    size = args.size
    batch = args.batch
    pack = 32
    if args.full==1:
        full=True
    else:
        full=False
    if args.binary==1:
        binary=True
    else:
        binary=False
        
    device = 0


    trainset, testset, classes = utils_own.load_dataset(dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                             shuffle=False, num_workers=2)
    
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if size==0:
        net = network.FC_small(full=full, binary=binary, first_sparsity=first_sparsity, rest_sparsity=rest_sparsity, align=True, ind=768, hid=128)
    elif size==1:
        net = network.FC_large(full=full, binary=binary, first_sparsity=first_sparsity, rest_sparsity=rest_sparsity, align=True, ind=768)
    elif size==2:
        net = network.CNN_tiny(full=full, binary=binary, conv_thres=first_sparsity, fc_thres=rest_sparsity, align=True)
    save_file = save_dir+'MNIST_s_{0}'.format(size)

    setup_logging(save_file + '_log.txt')
    logging.info("saving to %s", save_file)
    
    result_dic = save_file + '_result.pt'
    
    save_file += '.pt'
    
    if not args.cpu:
        torch.cuda.empty_cache()
        net.cuda(device)
    
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    lr_decay = np.power((2e-6/learning_rate), (1./100))

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)        
    
    ### Training section

    # Train without packing constraints
    utils_own.adjust_pack(net, 1)
    for epoch in range(0, 25):
        train_loss, train_prec1, train_prec5 = utils_own.train(trainloader, net, criterion, epoch, optimizer)
        val_loss, val_prec1, val_prec5 = utils_own.validate(testloader, net, criterion, epoch, verbal=True)
        scheduler.step()
    
    # Retrain with permutation + packing constraint
    utils_own.adjust_pack(net, pack)
    utils_own.permute_all_weights_once(net, pack=pack, mode=permute)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    for epoch in range(0, 25):
        train_loss, train_prec1, train_prec5 = utils_own.train(trainloader, net, criterion, epoch, optimizer)
        val_loss, val_prec1, val_prec5 = utils_own.validate(testloader, net, criterion, epoch, verbal=True)
        scheduler.step()

    # Fix pruned packs and fine tune
    for mod in net.modules():
        if hasattr(mod, 'mask'):
            mod.mask = torch.abs(mod.weight.data)
    net.pruned = True
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)
    best_prec1 = 0

    for epoch in range(0, 100):
        train_loss, train_prec1, train_prec5 = utils_own.train(trainloader, net, criterion, epoch, optimizer)
        val_loss, val_prec1, val_prec5 = utils_own.validate(testloader, net, criterion, epoch, verbal=False)
        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        if is_best:
            torch.save(net, save_file)
            logging.info('\n Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Training Prec {train_prec1:.3f} \t'
                         'Validation Loss {val_loss:.4f} \t'
                         'Validation Prec {val_prec1:.3f} \t'
                         .format(epoch+1, train_loss=train_loss, val_loss=val_loss,
                                 train_prec1=train_prec1, val_prec1=val_prec1))
        scheduler.step()
    logging.info('\nTraining finished!')

    # Extract data
    conv_count = 0
    bn2d_count = 0
    bn1d_count = 0
    fc_count = 0

    upload_dir = save_dir

    for mod in net.modules():
        if isinstance(mod, nn.Conv2d):
            print(mod)
            weight = mod.weight.data.type(torch.int16).cpu().numpy()
            conv_count += 1
            np.save(upload_dir+'conv_{0}_weight'.format(conv_count), weight)
        if isinstance(mod, nn.BatchNorm2d):
            print(mod)
            weight = mod.weight.data.cpu().numpy()
            bias = mod.bias.data.cpu().numpy()
            mean = mod.running_mean.cpu().numpy()
            var = mod.running_var.cpu().numpy()

            bn2d_count += 1
            np.save(upload_dir+'bn2d_{0}_weight'.format(bn2d_count), weight)
            np.save(upload_dir+'bn2d_{0}_bias'.format(bn2d_count), bias)
            np.save(upload_dir+'bn2d_{0}_mean'.format(bn2d_count), mean)
            np.save(upload_dir+'bn2d_{0}_var'.format(bn2d_count), var)

        if isinstance(mod, nn.Linear):
            print(mod)
            weight = mod.weight.data.type(torch.int16).cpu().numpy()
            fc_count += 1
            np.save(upload_dir+'fc_{0}_weight'.format(fc_count), weight)
        if isinstance(mod, nn.BatchNorm1d):
            print(mod)
            bn1d_count += 1
            if type(mod.weight) != type(None):
                weight = mod.weight.data.cpu().numpy()
                np.save(upload_dir+'bn1d_{0}_weight'.format(bn1d_count), weight)
            if type(mod.bias) != type(None):
                bias = mod.bias.data.cpu().numpy()
                np.save(upload_dir+'bn1d_{0}_bias'.format(bn1d_count), bias)
            mean = mod.running_mean.cpu().numpy()
            var = mod.running_var.cpu().numpy()

            np.save(upload_dir+'bn1d_{0}_mean'.format(bn1d_count), mean)
            np.save(upload_dir+'bn1d_{0}_var'.format(bn1d_count), var)
    return 0
        
if __name__ == '__main__':
    main()