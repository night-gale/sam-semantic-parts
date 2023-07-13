import cv2
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset


class PerSeg(Dataset):
    def __init__(self, root='./data/PerSeg', ref_idx=0):
        super().__init__()
        self.images_path = os.path.join(root, 'Images/')
        self.masks_path = os.path.join(root, 'Annotations/')
        self.ref_idx = ref_idx

        self.paths = []
        self.obj_names = []
        self.test_image_paths = []
        self.test_mask_paths = []
        self.cls = []

        for obj_name in os.listdir(self.images_path):
            if ".DS" in obj_name:
                continue
            self.paths.append(obj_name)
            test_images_path = os.path.join(self.images_path, obj_name)
            test_masks_path = os.path.join(self.masks_path, obj_name)
            for test_idx in range(len(os.listdir(test_images_path))):

                ref_image_path = os.path.join(self.images_path, obj_name, f'{self.ref_idx:02d}.jpg')
                ref_mask_path = os.path.join(self.masks_path, obj_name, f'{self.ref_idx:02d}.png')

                test_image_path = test_images_path + f'/{test_idx:02d}.jpg'
                test_mask_path = test_masks_path + f'/{test_idx:02d}.png'

                self.test_image_paths.append([ref_image_path, test_image_path])
                self.test_mask_paths.append([ref_mask_path, test_mask_path])
                self.cls.append(len(self.paths))
                self.obj_names.append(obj_name)

    def __len__(self):
        return len(self.test_image_paths)

    def __getitem__(self, index):
        ref_image_path, test_image_path = self.test_image_paths[index]
        ref_mask_path, test_mask_path = self.test_mask_paths[index]

        # Load images and masks
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

        # Load test image
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(test_mask_path)
        test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2RGB)

        ret = {
            'sup_msk': ref_mask,
            'sup_rgb': ref_image,
            'qry_msk': test_mask,
            'qry_rgb': test_image,
            'cls': self.cls[index],
            'obj_name': self.obj_names[index]
        }

        return ret

