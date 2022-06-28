import joblib
import numpy as np
import os
import torch

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class NuScenesDataset(Dataset):

    def __init__(self, root_dir, split='train', include_da=False, include_name=False, include_diff=True):
        '''
        root_dir (string): dir for the dataset (personal note: /local/DEEPLEARNING/nuscenes/preprocessed/v1.0-mini_6_3_local)
        split (string): train or test

        include_da: output the drivable area mask (useful in val or for training the DSF)
        '''
        assert split in ['train', 'val']
        self.split = split
        self.include_da = include_da
        self.include_name = include_name
        self.include_diff = include_diff

        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.da_dir = os.path.join(root_dir, 'drivable_area_mask')
        self.diff_dir = os.path.join(root_dir, 'diffmaps')
        self.traj_file = os.path.join(root_dir, 'gt_trajectories_%s.joblib' % split)

        assert os.path.exists(self.image_dir), 'missing images dir at %s' % self.image_dir
        assert os.path.exists(self.da_dir), 'missing drivable area dir at %s' % self.da_dir
        assert os.path.exists(self.traj_file), 'missing trajectories file at %s' % self.traj_file

        print(self.traj_file)

        self.load_gt_data()

        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

    def load_gt_data(self):
        '''
        Load the whole GT trajectories file (but not the images)
        '''
        with open(self.traj_file, 'rb') as gt_f:
            all_data = joblib.load(gt_f)

        # print(type(all_data))
        # print(len(all_data))
        # print(type(all_data[0]))
        # print(all_data[0])

        all_pasts, all_futures, all_imgpaths, all_isintersection = zip(*all_data)   # unzip (we get tuples)

        # from tuple to array with nice type stacking
        self.all_pasts = np.asarray(all_pasts)
        self.all_futures = np.asarray(all_futures)
        self.all_imgpaths = np.asarray(all_imgpaths)
        self.all_isintersection = np.asarray(all_isintersection)

        # print(self.all_pasts.shape)
        # print(self.all_futures.shape)
        # print(self.all_imgpaths.shape)
        self.past_shape = self.all_pasts.shape[1:]
        self.future_shape = self.all_futures.shape[1:]

        assert len(self.all_pasts) == len(self.all_futures) == len(self.all_imgpaths)
        self.nb_samples = len(self.all_pasts)

        print('loaded %s data of shape past %s future %s imgpaths %s' % (str(self.nb_samples), str(self.all_pasts.shape), str(self.all_futures.shape), str(self.all_imgpaths.shape)))
        # print(self.all_data[0])

    def check(self):
        print(type(self.all_pasts[0]))
        print(self.all_pasts[0].shape)
        print(self.all_pasts[0])

    def __len__(self):
        return self.nb_samples

    # def __getitem__(self, idx):
    #     print('dataset getitem idx (%s) %s' % (type(idx), str(idx)))
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()

    #     if isinstance(idx, int):
    #         idx = [idx]

    #     past = self.all_pasts[idx]
    #     future = self.all_futures[idx]

    #     assert past.shape == (len(idx), 12, 2), 'got shape %s' % str(past.shape)

    #     images = np.empty((len(idx), 500, 500, 3))   # TODO: img size should be in some variable
    #     for i, sample_index in enumerate(idx):
    #         img_name = os.path.join(self.image_dir, self.all_imgpaths[sample_index])
    #         image = io.imread(img_name)
    #         assert image.shape == (500, 500, 3)
    #         images[i] = image

    #     sample = {'past_xy': past, 'future_xy': future, 'image': images}

    #     return sample

    def __getitem__(self, idx):
        '''
        only one int idx at a time
        '''
        past = self.all_pasts[idx]
        future = self.all_futures[idx]

        assert past.shape == (12, 2), 'got shape %s' % str(past.shape)

        img_name = os.path.join(self.image_dir, self.all_imgpaths[idx])
        input_image = Image.open(img_name)

        input_tensor = self.preprocess(input_image)
        # input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        assert input_tensor.shape == (3, 224, 224), 'expected (224, 224, 3), got %s' % str(input_tensor.shape)

        sample = {'past_xy': past, 'future_xy': future, 'image': input_tensor, 'is_intersection': self.all_isintersection[idx]}

        # for val dir, also load the drivable area mask which is useful to compute diversity metrics
        if self.include_da:
            da_name = os.path.join(self.da_dir, 'da_' + self.all_imgpaths[idx])
            input_da = Image.open(da_name)

            da_tensor = self.preprocess(input_da)
            sample['da_mask'] = da_tensor

        if self.include_diff:
            d_name = os.path.join(self.diff_dir, 'diff_' + self.all_imgpaths[idx])
            input_d = Image.open(d_name)
            npd = np.asarray(input_d)
            npdt = torch.Tensor(npd)
            sample['d_mask'] = npdt

        if self.include_diff:
            dx_name = os.path.join(self.diff_dir, 'gx2_' + self.all_imgpaths[idx].replace('.png', '.npy'))
            with open(dx_name, 'rb') as f_dx:
                npdx = np.load(f_dx)

            npdxt = torch.Tensor(npdx)
            sample['dx_mask'] = npdxt

        if self.include_diff:
            dy_name = os.path.join(self.diff_dir, 'gy2_' + self.all_imgpaths[idx].replace('.png', '.npy'))
            with open(dy_name, 'rb') as f_dy:
                npdy = np.load(f_dy)

            npdyt = torch.Tensor(npdy)
            sample['dy_mask'] = npdyt

        # name is instance_sample
        if self.include_name:
            sample['name'] = self.all_imgpaths[idx].split('.')[0]

        return sample
