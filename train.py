import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import dice_score
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

from albumentations import (
    HorizontalFlip, Rotate, RandomCrop, GaussNoise, RandomBrightnessContrast, Compose,
    Resize, Normalize, OneOf, ElasticTransform, GridDistortion, Affine
)

def get_training_augmentation(input_height, input_width):
    train_transform = Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=90, p=0.5),
        RandomCrop(height=input_height, width=input_width, always_apply=True),
        GaussNoise(p=0.2),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            GridDistortion(p=0.5),
            Affine(scale=(0.5, 1.5), translate_percent=None, translate_px=None, rotate=None, shear=10, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
        ], p=1),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return train_transform

def get_validation_augmentation(input_height, input_width):
    val_transform = Compose([
        Resize(height=input_height, width=input_width),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return val_transform

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=1024, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=1024, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='DiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='DRAC2022/1.IMA/train',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='MultiStepLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-7, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='25', type=str)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    # resume
    parser.add_argument('--weights', default=None, type=str,
                        help='path to the model weights file (default: None)')
    
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            dice = dice_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            dice = dice_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('dice', avg_meters['dice'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice', avg_meters['dice'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                dice = dice_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                dice = dice_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()
    # Load model weights if specified
    if config['weights'] is not None and os.path.isfile(config['weights']):
        model.load_state_dict(torch.load(config['weights']))
        print("=> loaded weights from '{}'".format(config['weights']))
    else:
        print("=> no weights found at '{}'".format(config['weights']))

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2)

    config = vars(parse_args())
    input_h = config['input_h']
    input_w = config['input_w']

    train_transform = get_training_augmentation(input_h, input_w)
    val_transform = get_validation_augmentation(input_h, input_w)

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('dice', []),
        ('val_loss', []),
        ('val_dice', []),
    ])

    best_dice = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'MultiStepLR':
            pass
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        if epoch == config['epochs']/4:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['lr'] * 0.1
        current_lr = optimizer.param_groups[0]['lr']

        print('loss %.4f - dice %.4f - val_loss %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['dice'], val_log['loss'], val_log['dice']))

        log['epoch'].append(epoch)
        log['lr'].append(current_lr)
        log['loss'].append(train_log['loss'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_dice'].append(val_log['dice'])
        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        # Checkpoint saving every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = 'models/%s/Dice_model_epoch_%d_loss_%.4f_dice_%.4f.pth' % (config['name'], epoch + 1, val_log['loss'], val_log['dice'])
            torch.save(model.state_dict(), checkpoint_path)
            print("=> saved checkpoint for epoch %d with loss %.4f and dice %.4f" % (epoch + 1, val_log['loss'], val_log['dice']))

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'models/%s/model_epoch_%d_dice_%.4f.pth' %
                       config['name'], epoch + 1, val_log['dice'])
            best_dice = val_log['dice']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
