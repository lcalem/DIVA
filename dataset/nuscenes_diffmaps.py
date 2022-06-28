import argparse
import os
import re

import numpy as np

from PIL import Image
from scipy import ndimage


def create_diffmaps(path):
    count = 0

    da_dir = os.path.join(path, 'drivable_area_mask')
    out_dir = os.path.join(path, 'diffmaps')
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(da_dir):
        count += 1
        if count % 1000 == 0:
            print("done %s" % count)

        if not filename.endswith('.png'):
            raise

        img_path = os.path.join(da_dir, filename)
        img = Image.open(img_path)
        img = img.resize((224, 224))
        npimg = np.asarray(img)

        diffmap = ndimage.distance_transform_edt(npimg)

        # save diff image
        diff_path = os.path.join(out_dir, re.sub('^da_', 'diff_', filename))
        if not os.path.exists(diff_path):
            diff_im = Image.fromarray(diffmap)
            diff_im = diff_im.convert("L")
            diff_im.save(diff_path)

        # gradients (we load again because taking diffmap directly doesn't work for some reason)
        imgdiff = Image.open(diff_path)
        npimg = np.asarray(imgdiff)
        g = np.gradient(npimg)
        g2 = np.gradient(npimg, edge_order=2)
        gx = g[0]
        gy = g[1]

        gx2 = g2[0]
        gy2 = g2[1]

        gx_path = os.path.join(out_dir, re.sub('^da_', 'gx_', filename).replace('.png', '.npy'))
        if not os.path.exists(gx_path):
            with open(gx_path, 'wb') as f_gx:
                np.save(f_gx, gx)

        gy_path = os.path.join(out_dir, re.sub('^da_', 'gy_', filename).replace('.png', '.npy'))
        if not os.path.exists(gy_path):
            with open(gy_path, 'wb') as f_gy:
                np.save(f_gy, gy)

        gx2_path = os.path.join(out_dir, re.sub('^da_', 'gx2_', filename).replace('.png', '.npy'))
        if not os.path.exists(gx2_path):
            with open(gx2_path, 'wb') as f_gx2:
                np.save(f_gx2, gx2)

        gy2_path = os.path.join(out_dir, re.sub('^da_', 'gy2_', filename).replace('.png', '.npy'))
        if not os.path.exists(gy2_path):
            with open(gy2_path, 'wb') as f_gy2:
                np.save(f_gy2, gy2)


# python3 nuscenes_diffmaps.py -d /share/homes/lcalem/nuscenes/preprocessed/v1.0-trainval_6_3_full_reverse
# python3 nuscenes_diffmaps.py -d /share/homes/lcalem/nuscenes/preprocessed/v1.0-trainval_6_6_full_reverse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', required=True, help='path to the preprocessed dataset')

    args = parser.parse_args()

    create_diffmaps(args.datadir)



