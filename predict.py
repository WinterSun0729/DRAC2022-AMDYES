import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
# from sklearn.model_selection import train_train_split
from tqdm import tqdm

import archs
from dataset import Dataset, PreDataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import (
    HorizontalFlip, Rotate, RandomCrop, GaussNoise, RandomBrightnessContrast, Compose,
    Resize, Normalize, OneOf, ElasticTransform, GridDistortion, Affine
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs/DRAC2022/1.IMA/train', 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    # model.load_state_dict(torch.load('models/DRAC2022/1.IMA/train_NestedUNet_woDS/model_epoch_200_loss_0.6184_iou_0.3245.pth'))
    model.load_state_dict(torch.load('models/DRAC2022/1.IMA/train_NestedUNet_woDS/Dice_model_epoch_100_loss_0.7346_dice_0.2645.pth'))

    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    val_dataset = PreDataset(
        img_ids=img_ids,
        img_dir=os.path.join('inputs/DRAC2022/1.IMA/train/images'),
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                cv2.imwrite(os.path.join('inputs/DRAC2022/1.IMA/train/prediction', meta['img_id'][i] + '.png'), (output[i, c]*255).astype('uint8'))
    print('complete!')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
