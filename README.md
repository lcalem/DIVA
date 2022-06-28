


# Download & Preprocessing

1. Clone this repo in <REPO_ROOT>
2. Put <REPO_ROOT> in your `PYTHONPATH` (in your .bashrc or equivalent)
3. Install python packages in `requirements.txt`
4. Download nuscenes dataset & maps in <DATA_ROOT> according to [their website](https://www.nuscenes.org/download) (you need an account)
5. Download and install [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit). Note: if you have subsequent issues about not being able to import `nuscenes`, try putting the path of nuscenes-devkit at the start of your PYTHONPATH
6. Preprocess dataset (<REPO_ROOT>/dataset > `python3 preprocess_nuscenes.py v1.0-trainval <DATA_ROOT>`)

The resulting dataset should be in `<DATA_ROOT>/preprocessed` in the form of 2 .joblib files (train & val) and one `images` folder containing bird eye view inputs (names are instancetoken_sampletoken.jpeg).

Sanity check: there should be 16549 images both in `<DATA_ROOT>/preprocessed/v1.0-trainval_6_3_local/drivable_area_mask` and `<DATA_ROOT>/preprocessed/v1.0-trainval_6_3_local/images` (you can check with `ls -l | wc -l` in the folder)


# Training

1. Update the dataset path in `config/cvae_trainval.yaml`

2. In `<REPO_ROOT>/experiments`:

```
python3 main_pap.py -m train -a cvae_loc -o cvae_trainval -g 0 -n b-cvae_p_f --pretrained_enc --freeze_enc -b 10
```


# Eval


# Misc

## Frame of reference

The frame of reference for the bird eye view (BEV) is the following:
- the BEV is 50 x 50 meters in size
- the agent is at (0, 0) and this point is located such that:
    - there are 25m to the right and to the left of the vehicle (x axis)
    - there are 40m in front of the vehicle (towards the top of the image) and 10m behind the vehicle (bottom of the image) (y axis)