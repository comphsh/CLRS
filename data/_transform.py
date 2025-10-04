import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from numpy import random
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import transforms as T
from skimage.transform import resize
import abc

def build_transforms(transforms_):
    transforms = []
    # for transform in transforms_:
    #     Log.info("tranform={}".format(transform))
    #     Log.info("tran.kwargs={}".format(hasattr(transform,'kwargs')))
    #     for k,v in transform.items():
    #         Log.info("k={} v={}".format(k,v))
    #     Log.info("tran.kwargs={}".format(transform.kwargs))
    # for transform in transforms_:
    #     Log.info('tr.k.dict={}'.format(transform['kwargs']))
        # Log.info('tr.k.dict={}'.format(transform['kwargs'].__dict__))


    for transform in transforms_:
        # if hasattr(transform, 'kwargs') and transform.kwargs is not None:
            # kwargs = transform.kwargs.__dict__
            # transform = eval(f"{transform.name}")(**kwargs)
        if 'kwargs' in transform and transform['kwargs'] is not None:
            kwargs = transform['kwargs']
            transform = eval(f"{transform['name']}")(**kwargs)
        else:
            # transform = eval(f"{transform.name}()")
            transform = eval(f"{transform['name']}()")
        transforms.append(transform)
    return Compose(transforms)


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str

class MyCompose:

    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, data):

        for transform in self.transforms:
            data = transform(**data)
        return data

class batchgenerator_Compose(AbstractTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, **kwargs):
        for t in self.transforms:
            img = t(img, **kwargs)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor3D(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, with_sdf=False):
        img = torch.from_numpy(sample['image'])
        label = torch.from_numpy(sample['label'])
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(label.shape) == 3:
            label = label.unsqueeze(0)
        if with_sdf:
            sdf = torch.from_numpy(sample['sdf'])
            return {'image': img, 'label': label, 'sdf': sdf}

        return {'image': img, 'label': label}


# 3D transforms
class CenterCrop3D(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample, with_sdf=False):
        image, label = sample['image'], sample['label']
        if with_sdf:
            sdf = sample['sdf']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if with_sdf:
                for _ in range(sdf.shape[0]):
                    sdf[_] = np.pad(sdf[_], [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if with_sdf:
            for _ in range(sdf.shape[0]):
                sdf[_] = sdf[_][w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        return {'image': image, 'label': label}


class RandomCrop3D(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample, with_sdf=False):
        image, label = sample['image'], sample['label']
        if with_sdf:
            sdf = sample['sdf']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if with_sdf:
                temp_sdf = []
                for _ in range(sdf.shape[0]):
                    temp_sdf.append(np.pad(sdf[_], [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0))
                sdf = np.stack(temp_sdf)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        if with_sdf:
            sdf = sdf[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip3D(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample, with_sdf=False):
        image, label = sample['image'], sample['label']
        if with_sdf:
            sdf = sample['sdf']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        if with_sdf:
            sdf = np.rot90(sdf, k, axes=(1, 2))
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        if with_sdf:
            sdf = np.flip(sdf, axis=axis + 1).copy()
        if with_sdf:
            return {'image': image, 'label': label, 'sdf': sdf}
        return {'image': image, 'label': label}


class RandomNoise3D(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample, with_sdf=False):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * self.sigma,
                        2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        if with_sdf:
            return {'image': image, 'label': label, 'sdf': sample['sdf']}
        return {'image': image, 'label': label}


class CreateOnehotLabel3D(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


# Circle1
class RandomCrop3D_Circle1(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size=[80,160,160] , scale=True):  #[80,160,160]
        self.crop_d , self.crop_h , self.crop_w = output_size[0] , output_size[1] , output_size[2]
        self.scale = scale

    def __call__(self, image, label): # image=(4, 141, 131, 173)  label=(3, 141, 131, 173) output_size=[80,160,160]
        # randomly scale imgs
        # label=== (2, 128, 128, 128) (3, 128, 128, 128)
        # print("label===", image.shape, label.shape)
        [d0, d1, h0, h1, w0, w1] , [pd0,pd1,ph0,ph1,pw0,pw1] , scale_flag = self.locate_bbx_wScale(label)


        image = image[:, d0: d1, h0: h1, w0: w1]
        label = label[:, d0: d1, h0: h1, w0: w1]


        image = np.pad(image, [(0, 0), (pd0, pd1), (ph0, ph1), (pw0, pw1)], mode='constant', constant_values=0)
        label = np.pad(label, [(0, 0), (pd0, pd1), (ph0, ph1), (pw0, pw1)], mode='constant', constant_values=0)

        if scale_flag:
            if image.shape[1]!=self.crop_d or image.shape[2]!=self.crop_h or image.shape[3]!=self.crop_w:
                image = resize(image, (4, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0, clip=True, preserve_range=True)
                label = resize(label, (3, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True, preserve_range=True)

        return {'image': image, 'label': label}

    def locate_bbx_wScale(self, label):  #[3,155,240,240] , unique=[0,1]
        # randomly scale imgs
        scale_flag = False
        scaler = 1

        # if self.scale and np.random.uniform() < 0.5:
        #     scaler = np.random.uniform(0.9, 1.1)
        #     scale_flag = True

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        class_num, img_d, img_h, img_w = label.shape

        if random.random() < 0.5:
            selected_class = np.random.choice(class_num + 1)
            class_locs = []
            if selected_class != class_num:  #
                class_label = label[selected_class]
                class_locs = np.argwhere(class_label > 0)  #

            if selected_class == class_num or len(class_locs) == 0: #
                # if no foreground found, then randomly select
                #
                # NOTE:
                # a = np.random.randint(0,1) # A<= X < B
                # b = random.randint(1, 1) # # A<= X < =B
                d0 = random.randint(0, max(0 , img_d - scale_d))
                h0 = random.randint(0, max(0 , img_h - scale_h))
                w0 = random.randint(0, max(0 , img_w - scale_w))
                d1 = d0 + scale_d
                h1 = h0 + scale_h
                w1 = w0 + scale_w
                # print("not found this class!")
            else:
                selected_voxel = class_locs[np.random.choice(len(class_locs))]  #
                center_d, center_h, center_w = selected_voxel #

                d0 = center_d - scale_d // 2
                d1 = center_d + scale_d // 2

                h0 = center_h - scale_h // 2
                h1 = center_h + scale_h // 2

                w0 = center_w - scale_w // 2
                w1 = center_w + scale_w // 2
                # print("found this class!")
        else:
            d0 = random.randint(0, max(0, img_d - scale_d))
            h0 = random.randint(0, max(0, img_h - scale_h))
            w0 = random.randint(0, max(0, img_w - scale_w))

            d1 = d0 + scale_d
            h1 = h0 + scale_h
            w1 = w0 + scale_w
            # print("random upper crop!")

        if d0 < 0:
            pd0 = 0 - d0  # 填充0的数量
            d0 = 0
        else:
            pd0 = 0
        if d1 > img_d:
            pd1 = d1 - img_d
            d1 = img_d
        else:
            pd1 = 0
        # ---
        if h0 < 0:
            ph0 = 0 - h0
            h0 = 0
        else:
            ph0 = 0
        if h1 > img_h:
            ph1 = h1 - img_h
            h1 = img_h
        else:
            ph1 = 0
        # ---
        if w0 < 0:
            pw0 = 0 - w0
            w0 = 0
        else:
            pw0 = 0
        if w1 > img_w:
            pw1 = w1 - img_w
            w1 = img_w
        else:
            pw1 = 0
        # d0 = np.max([d0, 0])
        # d1 = np.min([d1, img_d])
        # h0 = np.max([h0, 0])
        # h1 = np.min([h1, img_h])
        # w0 = np.max([w0, 0])
        # w1 = np.min([w1, img_w])
        # print("crop_coord={}   paded_list={}".format([d0, d1, h0, h1, w0, w1]  , [pd0,pd1,ph0,ph1,pw0,pw1]))
        return [d0, d1, h0, h1, w0, w1] , [pd0,pd1,ph0,ph1,pw0,pw1] ,scale_flag


class RandomResize3D_Circle1(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size=[128,128,128] , scale=True):  #[80,160,160]
        self.crop_d , self.crop_h , self.crop_w = output_size[0] , output_size[1] , output_size[2]
        self.scale = scale

    def __call__(self, image, label): # image=(4, 141, 131, 173)  label=(3, 141, 131, 173) output_size=[80,160,160]
        if image.shape[1]!=self.crop_d or image.shape[2]!=self.crop_h or image.shape[3]!=self.crop_w:
            image = resize(image, (4, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0, clip=True, preserve_range=True)
            label = resize(label, (3, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True, preserve_range=True)

        return {'image': image, 'label': label}



# ShaSpec
class RandomCrop3D_SS(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size=[80,160,160] , scale=True):  #[80,160,160]
        self.crop_d , self.crop_h , self.crop_w = output_size[0] , output_size[1] , output_size[2]
        self.scale = scale



    def __call__(self, image, label):

        [d0, d1, h0, h1, w0, w1], scale_flag = self.locate_bbx_wScale(label)
        image = image[:, d0: d1, h0: h1, w0: w1]
        label = label[:, d0: d1, h0: h1, w0: w1]

        if scale_flag:
            if image.shape[1]!=self.crop_d or image.shape[2]!=self.crop_h or image.shape[3]!=self.crop_w:
                image = resize(image, (4, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0, clip=True, preserve_range=True)
                label = resize(label, (3, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True, preserve_range=True)

        return {'image': image, 'label': label}

    def locate_bbx_wScale(self, label):  #[3,155,240,240] , unique=[0,1]

        # randomly scale imgs
        scale_flag = False
        if self.scale and np.random.uniform() < 0.5:
            scaler = np.random.uniform(0.9, 1.1)
        # if self.scale and np.random.uniform() < 0.2:
        #     scaler = np.random.uniform(0.85, 1.25)
            scale_flag = True
        else:
            scaler = 1
        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        class_num, img_d, img_h, img_w = label.shape

        if random.random() < 0.5:
            selected_class = np.random.choice(class_num + 1)
            class_locs = []
            if selected_class != class_num:
                class_label = label[selected_class]
                class_locs = np.argwhere(class_label > 0)  #

            if selected_class == class_num or len(class_locs) == 0: #
                # if no foreground found, then randomly select
                d0 = random.randint(0, img_d - 0 - scale_d)
                h0 = random.randint(15, img_h - 15 - scale_h)
                w0 = random.randint(10, img_w - 10 - scale_w)
                d1 = d0 + scale_d
                h1 = h0 + scale_h
                w1 = w0 + scale_w
            else:
                selected_voxel = class_locs[np.random.choice(len(class_locs))]  #
                center_d, center_h, center_w = selected_voxel

                d0 = center_d - scale_d // 2
                d1 = center_d + scale_d // 2
                h0 = center_h - scale_h // 2
                h1 = center_h + scale_h // 2
                w0 = center_w - scale_w // 2
                w1 = center_w + scale_w // 2

                if h0 < 0: #
                    delta = h0 - 0
                    h0 = 0
                    h1 = h1 - delta
                if h1 > img_h:
                    delta = h1 - img_h
                    h0 = h0 - delta
                    h1 = img_h
                if w0 < 0:
                    delta = w0 - 0
                    w0 = 0
                    w1 = w1 - delta
                if w1 > img_w:
                    delta = w1 - img_w
                    w0 = w0 - delta
                    w1 = img_w
                if d0 < 0:
                    delta = d0 - 0
                    d0 = 0
                    d1 = d1 - delta
                if d1 > img_d:
                    delta = d1 - img_d
                    d0 = d0 - delta
                    d1 = img_d

        else:
            d0 = random.randint(0, img_d - 0 - scale_d)
            h0 = random.randint(15, img_h - 15 - scale_h)
            w0 = random.randint(10, img_w - 10 - scale_w)
            d1 = d0 + scale_d
            h1 = h0 + scale_h
            w1 = w0 + scale_w

        d0 = np.max([d0, 0])
        d1 = np.min([d1, img_d])
        h0 = np.max([h0, 0])
        h1 = np.min([h1, img_h])
        w0 = np.max([w0, 0])
        w1 = np.min([w1, img_w])

        return [d0, d1, h0, h1, w0, w1], scale_flag


class RandomMirror3D_SS(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self):
        pass

    def __call__(self, image , label):

        randi = np.random.rand(1)
        if randi <= 0.3:
            pass
        elif randi <= 0.4:
            image = image[:, :, :, ::-1]
            label = label[:, :, :, ::-1]
        elif randi <= 0.5:
            image = image[:, :, ::-1, :]
            label = label[:, :, ::-1, :]
        elif randi <= 0.6:
            image = image[:, ::-1, :, :]
            label = label[:, ::-1, :, :]
        elif randi <= 0.7:
            image = image[:, :, ::-1, ::-1]
            label = label[:, :, ::-1, ::-1]
        elif randi <= 0.8:
            image = image[:, ::-1, :, ::-1]
            label = label[:, ::-1, :, ::-1]
        elif randi <= 0.9:
            image = image[:, ::-1, ::-1, :]
            label = label[:, ::-1, ::-1, :]
        else:
            image = image[:, ::-1, ::-1, ::-1]
            label = label[:, ::-1, ::-1, ::-1]

        return {'image': image, 'label': label}

class Truncate3D_SS(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self ):
        pass

    def __call__(self, image):  #image , numpy
        MRI = image
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))  #


        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])  #。


        MRI[np.where(MRI >= idex_max)] = idex_max



        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)



        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        # np.where(condition, x, y)：：condition: 一个布尔数组，决定如何选择元素。x: 在 condition 为 True 时的值 。y: 在 condition 为 False的值。


        return MRI


class LabelToOnehot3D_BraTS2018(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self ):
        pass

    def __call__(self, label):  #numpy
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0, :, :, :] = np.where(ET, 1, 0)
        results_map[1, :, :, :] = np.where(WT, 1, 0)
        results_map[2, :, :, :] = np.where(TC, 1, 0)
        return results_map

class RandomGenerator(object):
    def __init__(self, output_size, p_flip=0.5, p_rot=0.5):
        self.output_size = output_size
        self.p_flip = p_flip
        self.p_rot = p_rot

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if torch.rand(1) < self.p_flip:
            image, label = self.random_rot_flip(image, label)
        elif torch.rand(1) < self.p_rot:
            image, label = self.random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8)).unsqueeze(0)
        sample['image'] = image
        sample['label'] = label
        return sample

    @staticmethod
    def random_rot_flip(image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    @staticmethod
    def random_rotate(image, label):
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label


class ToTensor:

    def __call__(self, sample):



        sample = {'image': F.to_tensor(sample['image']),
                  'label': torch.from_numpy(np.array(sample['label'])).unsqueeze(0)}

        # print("2 to tensor sample['image'] = {}  {}".format(np.array(sample['image']).shape, np.unique(sample['image'])))
        # print("2 to tensor sample['label'] = {}  {}".format(np.array(sample['label']).shape, np.unique(sample['label'])))
        # exit()
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToRGB:

    def __call__(self, sample):
        if sample['image'].shape[0] == 1:
            sample['image'] = sample['image'].repeat(3, 1, 1)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ConvertImageDtype(torch.nn.Module):

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, sample):
        sample['image'] = F.convert_image_dtype(sample['image'], self.dtype)
        sample['label'] = F.convert_image_dtype(sample['label'], self.dtype)
        return sample


class ToPILImage:

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, sample):
        sample['image'] = F.to_pil_image(sample['image'], self.mode)
        sample['label'] = F.to_pil_image(sample['label'], self.mode)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


# 2D transforms
class Normalize(T.Normalize):

    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)

    def forward(self, sample):
        sample['image'] = F.normalize(sample['image'], self.mean, self.std, self.inplace)
        return sample


class Resize(T.Resize):

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, interpolation)

    def forward(self, sample):
        sample['image'] = F.resize(sample['image'], self.size, self.interpolation)
        sample['label'] = F.resize(sample['label'], self.size, self.interpolation)
        # print("afsample['image'] = {}  {}".format(np.array(sample['image']).shape, np.unique(sample['image'])))  # [224，224，chnns]
        # print("af sample['label'] = {}  {}".format(np.array(sample['label']).shape, np.unique(sample['label'])))
        return sample


class CenterCrop(T.CenterCrop):

    def __init__(self, size):
        super().__init__(size)

    def forward(self, sample):
        sample['image'] = F.center_crop(sample['image'], self.size)
        sample['label'] = F.center_crop(sample['label'], self.size)
        return sample


class Pad(T.Pad):

    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__(padding, fill, padding_mode)

    def forward(self, sample):
        sample['label'] = F.pad(sample['image'], self.padding, self.fill, self.padding_mode)
        sample['label'] = F.pad(sample['label'], self.padding, self.fill, self.padding_mode)
        return sample


class RandomCrop(T.RandomCrop):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def forward(self, sample):
        img = sample['image']
        label = sample['label']
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        sample['image'] = F.crop(img, i, j, h, w)
        sample['label'] = F.crop(label, i, j, h, w)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class RandomFlip(torch.nn.Module):

    def __init__(self, p=0.5, direction='horizontal'):
        super().__init__()
        assert 0 <= p <= 1
        assert direction in ['horizontal', 'vertical', None], 'direction should be horizontal, vertical or None'
        self.p = p
        self.direction = direction

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img, label = sample['image'], sample['label']
            if self.direction == 'horizontal':
                sample['image'] = F.hflip(img)
                sample['label'] = F.hflip(label)
            elif self.direction == 'vertical':
                sample['image'] = F.vflip(img)
                sample['label'] = F.vflip(label)
            else:
                if torch.rand(1) < 0.5:
                    sample['image'] = F.hflip(img)
                    sample['label'] = F.hflip(label)
                else:
                    sample['image'] = F.vflip(img)
                    sample['label'] = F.vflip(label)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(T.RandomResizedCrop):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)


    def forward(self, sample):
        img, mask = sample['image'], sample['label']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        sample['image'] = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        sample['label'] = F.resized_crop(mask, i, j, h, w, self.size, self.interpolation)
        return sample


class RandomRotation(T.RandomRotation):

    def __init__(
            self,
            degrees,
            interpolation=InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=0,
            p=0.5,
            resample=None
    ):
        super().__init__(degrees, interpolation, expand, center, fill)
        self.p = p

    def forward(self, sample):
        if torch.rand(1) > self.p:
            return sample
        img, label = sample['image'], sample['label']
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        label_fill = self.fill
        if isinstance(label, torch.Tensor):
            if isinstance(label_fill, (int, float)):
                label_fill = [float(label_fill)] * F.get_image_num_channels(label)
            else:
                label_fill = [float(f) for f in label_fill]
        angle = self.get_params(self.degrees)
        sample['image'] = F.rotate(img, angle, InterpolationMode.BILINEAR  ,self.expand, self.center, fill)
        sample['label'] = F.rotate(label, angle, InterpolationMode.NEAREST ,self.expand, self.center, label_fill)
        return sample


class RandomRotation90(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, sample):

        if torch.rand(1) < self.p:
            rot_times = random.randint(0, 4)
            sample['image'] = torch.rot90(sample['image'], rot_times, [1, 2])
            sample['label'] = torch.rot90(sample['label'], rot_times, [1, 2])
        return sample


class RandomErasing(T.RandomErasing):

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__(p, scale, ratio, value, inplace)

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img, label = sample['image'], sample['label']
            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value, ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    "{} (number of input channels)".format(img.shape[-3])
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            sample['image'] = F.erase(img, x, y, h, w, v, self.inplace)
            sample['label'] = F.erase(label, x, y, h, w, v, self.inplace)
        return sample


class GaussianBlur(T.GaussianBlur):

    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=0.5):
        super().__init__(kernel_size, sigma)
        self.p = p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            sample['image'] = F.gaussian_blur(sample['image'], self.kernel_size, [sigma, sigma])
        return sample


class RandomGrayscale(T.RandomGrayscale):

    def __init__(self, p=0.1):
        super().__init__(p)

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img = sample['image']
            if len(img.shape) == 4:
                img = img.permute(3, 0, 1, 2).contiguous()
                if img.size(1) == 1:
                    img = img.repeat(1, 3, 1, 1)
            num_output_channels = F.get_image_num_channels(img)
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
            if len(img.shape) == 4:
                img = img.permute(1, 2, 3, 0).contiguous()
                img = img[0].unsqueeze(0)
            sample['image'] = img
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)


class ColorJitter(T.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.):
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img = sample['image']
            if len(img.shape) == 4:
                img = img.permute(3, 0, 1, 2).contiguous()
            elif img.size(0) == 1:
                img = img.repeat(3, 1, 1)
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)

            if len(img.shape) == 4:
                img = img.permute(1, 2, 3, 0).contiguous()
                img = img[0].unsqueeze(0)
            sample['image'] = img
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
