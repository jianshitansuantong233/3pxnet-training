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

parser = argparse.ArgumentParser(description='PyTorch CNN Training')

parser.add_argument('--save_dir', metavar='SAVE_DIR', default='./training_data/conv/', help='Save dir')
parser.add_argument('--batch', metavar='BATCH', default=256, type=int, help='Batch size to use')
parser.add_argument('--size', metavar='SIZE', default=0, type=int, help='Size of network. 1 means large and 0 means medium')
parser.add_argument('--dataset', metavar='DATASET', default='SVHN', help='Dataset to train on. Currently choose from SVHN and CIFAR10')
parser.add_argument('--full', metavar='FULL', default=0, type=int, help='If 1, train with full precision')
parser.add_argument('--binary', metavar='BINARY', default=0, type=int, help='If 1, train with dense binary')
parser.add_argument('--conv_sparsity', metavar='CONV_SPARSITY', default=0.9, type=float, help='Sparsity of the convolutional layers')
parser.add_argument('--fc_sparsity', metavar='FC_SPARSITY', default=0.9, type=float, help='Sparsity of fully-connected layers')
parser.add_argument('--permute', metavar='PERMUTE', default=-1, type=int, help='Permutation method. -1 means no permutation')
parser.add_argument('--pad', metavar='PAD', default=0, type=int, help='Padding value used')
parser.add_argument('--seed', metavar='SEED', default=0, type=int, help='Seed to use for this run')
parser.add_argument('--cpu', dest='cpu', default=False, action='store_true', help='Train using cpu')

def main():
    global args, best_prec1
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    dataset = args.dataset
    conv_sparsity = args.conv_sparsity
    fc_sparsity = args.fc_sparsity
    permute = args.permute
    size = args.size
    batch = args.batch
    pad = args.pad
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
    if args.size==1:    
        net = network.CNN_large(full=full, binary=binary, conv_thres=conv_sparsity, fc_thres=fc_sparsity, align=True, pad=pad)
    elif args.size==0:
        net = network.CNN_medium(full=full, binary=binary, conv_thres=conv_sparsity, fc_thres=fc_sparsity, align=True, pad=pad)
        
    save_file = save_dir + '{0}_s_{1}'.format(dataset, size)

    setup_logging(save_file + '_log.txt')
    logging.info("saving to %s", save_file)
    
    save_file += '.pt'
    regime = getattr(net, 'regime', {0: {'optimizer': 'Adam', 'lr': 0.1,'momentum': 0.9,'weight_decay': 1e-4}})  
    
    if not args.cpu:
        torch.cuda.empty_cache()
        net.cuda(device)
    
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    lr_decay = np.power((2e-6/learning_rate), (1./100))

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    ### Training section
    
    # Train without packing constraints
    utils_own.adjust_pack(net,1)
    for epoch in range(0, 25):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
        train_loss, train_prec1, train_prec5 = utils_own.train(trainloader, net, criterion, epoch, optimizer)
        val_loss, val_prec1, val_prec5 = utils_own.validate(testloader, net, criterion, epoch, verbal=True)
    
    # Retrain with permutation + packing constraint
    utils_own.adjust_pack(net, pack)
    utils_own.permute_all_weights_once(net, pack=pack, mode=permute)

    for epoch in range(0, 25):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
        train_loss, train_prec1, train_prec5 = utils_own.train(trainloader, net, criterion, epoch, optimizer)
        val_loss, val_prec1, val_prec5 = utils_own.validate(testloader, net, criterion, epoch, verbal=True)
    
    # Fix pruned packs and fine tune
    for mod in net.modules():
        if hasattr(mod, 'mask'):
            mod.mask = torch.abs(mod.weight.data)    
    net.pruned = True
    best_prec1 = 0
    
    for epoch in range(0, 200):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
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
    logging.info('\nTraining finished')
    
    # Extract data
    conv_count = 0
    bn2d_count = 0
    bn1d_count = 0
    fc_count = 0

    upload_dir = save_dir

    if args.size==1:
        x=Variable(torch.randn(1,3,32,32,requires_grad=True,device='cuda'))
        torch_out=net(x)
        torch.onnx.export(net,x,"training_data/CNN_Large.onnx",verbose=True,opset_version=9,input_names = ['input'], output_names = ['output'])
    elif args.size==0:
        x=Variable(torch.randn(1,3,32,32,requires_grad=True,device='cuda'))
        torch_out=net(x)
        torch.onnx.export(net,x,"training_data/CNN_Medium.onnx",verbose=True,opset_version=9,input_names = ['input'], output_names = ['output'])

if __name__ == '__main__':
    main()
