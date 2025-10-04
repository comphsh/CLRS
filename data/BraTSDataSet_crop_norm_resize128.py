import os.path as osp
import numpy as np
import random
from torch.utils import data
import nibabel as nib
from skimage.transform import resize
import math
import os
import ipdb
import time
# from batchgenerators.transforms import Compose
from data._transform import MyCompose, batchgenerator_Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform

from data._transform import RandomResize3D_Circle1 , RandomMirror3D_SS , Truncate3D_SS , LabelToOnehot3D_BraTS2018

def get_train_transform(patch_size):
    tr_transforms = []

    tr_transforms.append(
        SpatialTransform(
            patch_size,
            patch_center_dist_from_border=[i // 2 for i in patch_size],
            do_elastic_deform=True,
            alpha=(0., 900.),
            sigma=(9., 13.),
            do_rotation=True,
            angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            do_scale=True, scale=(0.85, 1.25),
            border_mode_data='constant', border_cval_data=0,
            order_data=3, border_mode_seg="constant", border_cval_seg=-1,
            order_seg=1,
            random_crop=True,
            p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,
            data_key="image", label_key="label")
    )
    # tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    # tr_transforms.append( GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5, p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    # tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
    #                                    order_upsample=3, p_per_sample=0.25,
    #                                    ignore_axes=None, data_key="image"))

    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="image"))

    # tr_transforms.append(MirrorTransform(axes=(0, 1, 2), data_key="image", label_key="label"))

    # now we compose these transforms together
    tr_transforms = batchgenerator_Compose(tr_transforms)
    return tr_transforms


def my_collate(batch):
    image, label = zip(*batch)
    # (1, 4, 80, 160, 160) (1, 3, 80, 160, 160)
    # print(np.array(image).shape , np.array(label).shape)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    # print(image.shape, label.shape)  #(1, 4, 80, 160, 160) (1, 3, 80, 160, 160)

    data_dict = {'image': image, 'label':label}
    tr_transforms = get_train_transform(patch_size=image.shape[2:])
    data_dict = tr_transforms(**data_dict)
    return data_dict


class BraTSDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(128, 128, 128), scale=True, mirror=True, ignore_label=255, proj_path=''):
        self.data_root = root
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.crop_size = crop_size

        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror

        self.train_transforms = []
        self.train_transforms.append(RandomResize3D_Circle1(output_size=crop_size, scale=True))
        self.train_transforms.append(RandomMirror3D_SS())
        self.train_transforms = MyCompose(self.train_transforms)

        print("root={} listpath={}".format( root , list_path))
        print("projpath/listpath={}/{}".format(proj_path, self.list_path))
        self.img_ids= []
        for dirname in open(os.path.join(proj_path, self.list_path)): #
            if 'HGG' in dirname:
                self.img_ids.append( os.path.join(self.data_root , 'HGG' , dirname.strip().replace('HGG_' , '')))
            elif 'LGG' in dirname:
                self.img_ids.append(os.path.join(self.data_root, 'LGG', dirname.strip().replace('LGG_', '')))
            else:
                exit(-1)

        self.files = []
        for item in self.img_ids:

            filepath = item + '/' +  osp.basename(item)  #
            flair_file = filepath + '_flair.nii'
            t1_file = filepath + '_t1.nii'
            t1ce_file = filepath + '_t1ce.nii'
            t2_file = filepath + '_t2.nii'

            label_file = filepath + '_seg_onehot.npy'
            name = osp.splitext(osp.basename(filepath))[0]
            # print("naem= {}".format(name))
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "name": name
            })

        print(self.img_ids)
        print('TrainSet {} images( repeated patients ? ) are loaded!'.format(len(self.img_ids)))

        self.image_list = []
        self.label_list = []
        self.fileid_list = {}
        self.pointer = 0

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):  # for locate bbx with scale
        # tt1 = time.time()
        if index not in self.fileid_list:
            datafiles = self.files[index]

            flairNII = nib.load(datafiles["flair"])
            t1NII = nib.load(datafiles["t1"])
            t1ceNII = nib.load(datafiles["t1ce"])
            t2NII = nib.load(datafiles["t2"])
            image = np.array([flairNII.get_fdata(), t1NII.get_fdata(), t1ceNII.get_fdata(), t2NII.get_fdata()])
            label = np.load(datafiles['label'])

            image = image.astype(np.float32)
            label = label.astype(np.float32)

            image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
            label = label.transpose((0, 3, 1, 2))  # Depth x H x W


            if image.shape[1] < self.crop_size[0] or image.shape[2] < self.crop_size[1] or image.shape[3] < self.crop_size[2]:
                pd_left = (self.crop_size[0] - image.shape[1]) // 2
                pd_right = (self.crop_size[0] - image.shape[1]) - pd_left
                ph_left = (self.crop_size[1] - image.shape[2]) // 2
                ph_right = (self.crop_size[1] - image.shape[2]) - ph_left
                pw_left = (self.crop_size[2] - image.shape[3]) // 2
                pw_right = (self.crop_size[2] - image.shape[3]) - pw_left
                image = np.pad(image, [(0, 0), (pd_left, pd_right), (ph_left, ph_right), (pw_left, pw_right)],  mode='constant', constant_values=0)
                label = np.pad(label, [(0, 0), (pd_left, pd_right), (ph_left, ph_right), (pw_left, pw_right)],  mode='constant', constant_values=0)

            self.image_list.append(image)
            self.label_list.append(label)
            self.fileid_list[index] = self.pointer
            self.pointer += 1

        else:
            image = self.image_list[self.fileid_list[index]]
            label = self.label_list[self.fileid_list[index]]


        sample = {'image' : image , 'label' : label}
        sample = self.train_transforms(sample)

        image = sample['image']
        label = sample['label']

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image, label


class BraTSValDataSet(data.Dataset):
    def __init__(self, root="", list_path='' , crop_size=(128, 128, 128), proj_path=''):
        self.data_root = root
        self.list_path = list_path
        self.crop_size=crop_size
        self.is_transpose = False

        print( "Valset path = ", root , list_path)
        self.img_ids = []

        print("projpath/listpath={}/{}".format(proj_path, self.list_path))
        for dirname in open(os.path.join(proj_path, self.list_path)):
            if 'HGG' in dirname:
                self.img_ids.append(os.path.join(self.data_root, 'HGG', dirname.strip().replace('HGG_', '')))
            elif 'LGG' in dirname:
                self.img_ids.append(os.path.join(self.data_root, 'LGG', dirname.strip().replace('LGG_', '')))
            else:
                exit(-1)



        self.files = []
        for item in self.img_ids:
            filepath = item + '/' + osp.basename(item)

            flair_file = filepath + '_flair.nii'
            t1_file = filepath + '_t1.nii'
            t1ce_file = filepath + '_t1ce.nii'
            t2_file = filepath + '_t2.nii'

            label_file = filepath + '_seg_onehot.npy'
            affine_file = filepath + '_affine.npy'

            cropsize_file = filepath + '_crop_size.npy'
            name = osp.splitext(osp.basename(filepath))[0]
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "affine": affine_file,
                "cropsize": cropsize_file,
                "name": name,
            })
        print('ValSet {} images(not repreated) are loaded!'.format(len(self.img_ids)))

        self.image_list = []
        self.label_list = []
        self.affine_list = []
        self.cropsize_list = []
        self.name_list = []
        self.fileid_list = {}
        self.pointer = 0

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        if index not in self.fileid_list:
            datafiles = self.files[index]
            flairNII = nib.load(datafiles["flair"])
            t1NII = nib.load(datafiles["t1"])
            t1ceNII = nib.load(datafiles["t1ce"])
            t2NII = nib.load(datafiles["t2"])
            image = np.array([flairNII.get_data(), t1NII.get_data(), t1ceNII.get_data(), t2NII.get_data()])
            label = np.load(datafiles['label'])
            name = datafiles["name"]

            affine = flairNII.affine
            cropsize = np.load(datafiles['cropsize'])

            image = image.transpose((0, 3, 1, 2))
            label = label.transpose((0, 3, 1, 2))
            self.is_transpose = True

            if image.shape[1] < self.crop_size[0] or image.shape[2] < self.crop_size[1] or image.shape[3] < self.crop_size[2]:
                pd_left = (self.crop_size[0] - image.shape[1]) // 2
                pd_right = (self.crop_size[0] - image.shape[1]) - pd_left
                ph_left = (self.crop_size[1] - image.shape[2]) // 2
                ph_right = (self.crop_size[1] - image.shape[2]) - ph_left
                pw_left = (self.crop_size[2] - image.shape[3]) // 2
                pw_right = (self.crop_size[2] - image.shape[3]) - pw_left
                image = np.pad(image, [(0, 0), (pd_left, pd_right), (ph_left, ph_right), (pw_left, pw_right)],
                               mode='constant', constant_values=0)
                label = np.pad(label, [(0, 0), (pd_left, pd_right), (ph_left, ph_right), (pw_left, pw_right)], mode='constant', constant_values=0)

            self.image_list.append(image)
            self.label_list.append(label)
            self.affine_list.append(affine)
            self.cropsize_list.append(cropsize)
            self.name_list.append(name)
            self.fileid_list[index] = self.pointer
            self.pointer += 1
        else:
            image = self.image_list[self.fileid_list[index]]
            label = self.label_list[self.fileid_list[index]]
            affine = self.affine_list[self.fileid_list[index]]
            cropsize = self.cropsize_list[self.fileid_list[index]]
            name = self.name_list[self.fileid_list[index]]


        return image, label, name, affine, np.array(cropsize)



