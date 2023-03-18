import xml.dom.minidom as xmldom
import cv2
import mmcv
import numpy as np
from datasets.make_shrink_map import *
from datasets.augs import *
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
import torch
import os
from utils.utilers import get_config

config = get_config('./pan-config.yaml')
if config['in_colab'] == True:
    ctw_root_dir = '/content/ctw-1500'
else:
    ctw_root_dir = '/Users/starfish/Desktop/datasets/ctw-1500'
ctw_train_data_dir = os.path.join(ctw_root_dir, 'train_images')
ctw_train_gt_dir = os.path.join(ctw_root_dir, 'train_labels')

ctw_test_data_dir = 'test_samples/imgs/'
ctw_test_gt_dir = 'test_samples/gt/'


def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception:
        print(img_path)
        raise
    return img


def get_ann(img, gt_path):
    domobj = xmldom.parse(gt_path)
    h, w = img.shape[0:2]
    w_h = np.array([w, h]).astype(np.float32)
    elementobj = domobj.documentElement
    # labels = elementobj.getElementsByTagName('label')
    segs = elementobj.getElementsByTagName('segs')
    bboxes = []
    words = []
    for i in range(len(segs)):
        b = segs[i].firstChild.data.split(',')
        b = [int(x) for x in b]
        b = np.array(b, dtype=np.float32).reshape(-1, 2)
        b /= w_h
        bboxes.append(b.reshape(-1))
        words.append('???')

    return bboxes, words



class PAN_CTW(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=640,
                 kernel_scale=0.7,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (
            img_size is None or isinstance(img_size, tuple)) else (img_size,
                                                                   img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.read_type = read_type

        if split == 'train':
            data_dirs = [ctw_train_data_dir]
            gt_dirs = [ctw_train_gt_dir]
        elif split == 'test':
            data_dirs = [ctw_test_data_dir]
            gt_dirs = [ctw_test_gt_dir]
        else:
            print('Error: split must be train or test!')
            raise

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = [
                img_name for img_name in mmcv.utils.scandir(data_dir, '.jpg')
            ]
            img_names.extend([
                img_name for img_name in mmcv.utils.scandir(data_dir, '.png')
            ])

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = os.path.join(data_dir, img_name)
                # img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = img_name.split('.')[0] + '.xml'
                gt_path = os.path.join(gt_dir, gt_name)
                # gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

        if report_speed:
            target_size = 3000
            data_size = len(self.img_paths)
            extend_scale = (target_size + data_size - 1) // data_size
            self.img_paths = (self.img_paths * extend_scale)[:target_size]
            self.gt_paths = (self.gt_paths * extend_scale)[:target_size]

        self.max_word_num = 200

    def __len__(self):
        return len(self.img_paths)

    def prepare_train_data(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann(img, gt_path)

        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]

        if self.is_transform:
            img = random_scale(img, self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                bboxes[i] = np.reshape(
                    bboxes[i] * ([img.shape[1], img.shape[0]] *
                                 (bboxes[i].shape[0] // 2)),
                    (bboxes[i].shape[0] // 2, 2)).astype('int32')
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[
                1], imgs[2], imgs[3:]

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255,
                                         saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
        )

        return data

    def prepare_test_data(self, index):
        img_path = self.img_paths[index]
        # print(img_path)
        img = get_img(img_path, self.read_type)
        img_meta = dict(org_img_size=np.array(img.shape[:2]))
        img_name = img_path.split("/")[-1]
        img_meta.update(dict(img_path=img_path))
        img_meta.update(dict(img_name=img_name))

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(img_size=np.array(img.shape[:2])))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)

        data = dict(imgs=img, img_metas=img_meta)

        return data

    def __getitem__(self, index):
        if self.split == 'train':
            return self.prepare_train_data(index)
        elif self.split == 'test':
            return self.prepare_test_data(index)
 