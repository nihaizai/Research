import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import datasets
import models
from utils import AverageMeter, Logger
from center_loss import CenterLoss

parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='gei', choices=['mnist','gei'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.1, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=4000)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='cnn')
# misc
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")

args = parser.parse_args()



def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

#==================================dataset loading============================
    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers,
    )

    trainloader, testloader = dataset.trainloader, dataset.testloader

    print("Creating model: {}".format(args.model))
    model = models.create(name=args.model, num_classes=dataset.num_classes)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=dataset.num_classes, feat_dim=2, use_gpu=use_gpu)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()
  
    xent_plot = []
    cent_plot = []
    loss_plot = []
    
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        xent_losses,cent_losses,losses = train(model, criterion_xent, criterion_cent,
                                               optimizer_model, optimizer_centloss,
                                               trainloader, use_gpu, dataset.num_classes, epoch)
        xent_plot.append(xent_losses.avg)
        cent_plot.append(cent_losses.avg)
        loss_plot.append(losses.avg) 
        if args.stepsize > 0: scheduler.step()

#        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
#            print("==> Test")
#            acc, err = test(model, testloader, use_gpu, dataset.num_classes, epoch)
#            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
        
        
        if epoch % 100 == 0:
            state = {
                    'cnn':model.state_dict()
                    }
            torch.save(state,'/home/mg/code/GEI+PTSN/train/pytorch-center-loss-master/snapshots_512/snapshot_%d.t7'  %epoch)
            print('model save at epoch %d' % epoch)
    
    plot_losses(xent_plot,cent_plot,loss_plot,prefix='losses')
    

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(model, criterion_xent, criterion_cent,
          optimizer_model, optimizer_centloss,
          trainloader, use_gpu, num_classes, epoch):
    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()
    
    
    if args.plot:
        all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features1,features2, outputs = model(data)
        #print("features shape:{}".format(features.shape))
        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features2, labels)
        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()
        
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))

        if args.plot:
            if use_gpu:
                all_features.append(features2.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
                #print("all_features:{}".format(len(all_features)))
                #print("all_labels:{}".format(len(all_labels)))
            else:
                all_features.append(features2.data.numpy())
                all_labels.append(labels.data.numpy())
        
     
          
        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg, cent_losses.val, cent_losses.avg))

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix='train')
    return xent_losses,cent_losses,losses

def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            features, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
            
            if args.plot:
                if use_gpu:
                    all_features.append(features.data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                else:
                    all_features.append(features.data.numpy())
                    all_labels.append(labels.data.numpy())

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix='test')

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

def plot_losses(xent_plot,cent_plot,loss_plot,prefix='losses'):
    ax1 = plt.subplot(3,1,1)
    ax1.set_title('xent_loss')
    plt.plot(np.arange(len(xent_plot)),xent_plot)
    
    ax2 = plt.subplot(3,1,2)
    ax2.set_title('cent_loss')
    plt.plot(np.arange(len(cent_plot)),cent_plot)
    
    ax3 = plt.subplot(3,1,3)
    ax3.set_title('total_loss')
    plt.plot(np.arange(len(loss_plot)),loss_plot)
    
    plt.tight_layout()
    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'losses.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
    
    










def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
              [0.1,0.2,0.3],[0.3,0.4,0.5],[0.6,0.7,0.8],[0.9,0.1,0.2],[0.8,0.4,0.6], 
              [0.7,0.3,0.5],[0.6,0.2,0.7],[0.5,0.5,0.5],[0.4,0.6,0.7],[0.3,0.7,0.9],
              [0.2,0.6,0.8],[0.1,0.5,0.5],[0.2,0.4,0.6],[0.3,0.3,0.3],[0.4,0.2,0.1],
              [0.5,0.3,0.2],[0.6,0.3,0.1],[0.7,0.8,0.1],[0.8,0.1,0.3],[0.9,0.6,0.5],
              [0.1,0.7,0.9],[0.2,0.3,0.5],[0.3,0.5,0.6],[0.4,0.4,0.3],[0.5,0.8,0.3], 
              [0.6,0.5,0.4],[0.7,0.1,0.8],[0.8,0.8,0.8],[0.9,0.3,0.7],[0.1,0.9,0.4],
              [0.2,0.8,0.2],[0.3,0.2,0.1],[0.4,0.8,0.5],[0.5,0.6,0.4],[0.6,0.4,0.3],
              [0.7,0.5,0.4],[0.8,0.3,0.5],[0.9,0.4,0.4],[0.1,0.6,0.8],[0.2,0.7,0.9],
              [0.3,0.6,0.4],[0.4,0.1,0.2],[0.5,0.4,0.7],[0.6,0.6,0.6],[0.7,0.7,0.7], 
              [0.8,0.5,0.1],[0.9,0.3,0.3],[0.1,0.7,0.2],[0.2,0.2,0.2],[0.3,0.3,0.3],
              [0.4,0.4,0.4],[0.5,0.5,0.5]
              ]
#    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
#              'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
#              'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
#              'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
#              'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
#              'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
#              'C0', 'C1'
#            ]
#    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        #print(len(features[labels==label_idx]))
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                '40', '41', '42', '43', '44', '45', '46', '47','48', '49',
                '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
                '60', '61'
                ], loc='upper right')
#        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
#                    ], loc='upper right')
    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()





