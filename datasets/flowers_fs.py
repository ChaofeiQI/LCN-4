import os, random, pickle, math
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .z_samplers import register

@register('flowers-fs')  # 71-15-16
class Flowers_FS(Dataset):
    def __init__(self, root_path, split='train', **kwargs):
        # 读取文件信息
        split_tag = split
        split_file = 'flowers_fs_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']
        
        data = [Image.fromarray(x) for x in data]  # FIXME: Need to check the image size.
        old_classes = np.unique(label)
        old2new_class = {old_classes[i]:i for i in range(len(old_classes))}
        new_label = [old2new_class[l] for l in label]
        self.data = data
        self.label = new_label
        self.n_classes = max(self.label) + 1

        # Augmentation settings. Note: Here we use 'aug' instead of 'augment' in original repo.
        aug = kwargs.get('aug')

        if aug == 'long':    # (similar to) Xiaolong's settings, which is referred as 'original repo'.
            assert False 
            image_size = 80  # Note: This is different from original repo, which sets image_size = 80.
            norm_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
            normalize = transforms.Normalize(**norm_params)

            self.default_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,  
            ])

            if split=='train':
                self.transform = transforms.Compose([    # Note: This is equivalent to augment == 'resize' in original repo. The setting with augment == 'crop' is removed here.
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = self.default_transform  # Note: In original repo, it is used when augment settings is None.

            def convert_raw(x):
                mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
                std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
                return x * std + mean
            self.convert_raw = convert_raw

        elif aug=='lee':  # Kwonjoon Lee's settings.
            mean_pix = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
            std_pix = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
            normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
            
            if split=='train' :
                self.transform = transforms.Compose([
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    # lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = transforms.Compose([
                    # lambda x: np.asarray(x),
                    transforms.Resize((84, 84), interpolation=3),
                    transforms.ToTensor(),
                    normalize
                ])

        elif aug == 'lee-test': # Kwonjoon Lee's settings: Always use the test settings.
            mean_pix = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
            std_pix = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
            normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            raise ValueError()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img
