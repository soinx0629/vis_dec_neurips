import nibabel as nib
import numpy as np
import hcp_utils as hcp
import os

# SUBJECT_LIST_DIR = './YourSubList/'
HCP_DATA_DIR = 'your path to HCP data, by default HCP_DATA_DIR/npz'
ROOT_DIR = 'your path to save preprocessed data'
# IMG_PATH = 'MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'
# IMG_PATH = 'rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'
IMG_PATH = 'rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'


subjects = []
with open('ls_hcp1200', 'r') as pp:
    for line in pp:
        line_spl = line.split(' ')[-1].strip('\n').replace('/', '')
        if len(line_spl)>2:
            subjects.append(line_spl)


v1_idx = np.where(((hcp.mmp.map_all == 1)) | (hcp.mmp.map_all == 181))[0] 
v2_idx = np.where(((hcp.mmp.map_all == 4)) | (hcp.mmp.map_all == 184))[0]
v3_idx = np.where(((hcp.mmp.map_all == 5)) | (hcp.mmp.map_all == 185))[0]
v4_idx = np.where(((hcp.mmp.map_all == 6)) | (hcp.mmp.map_all == 186))[0]

# sub_list = [sub.split('_')[0] for sub in os.listdir(SUBJECT_LIST_DIR) if '.txt' in sub]
subjects = []
for sub in subjects:
    print('sub is', sub)
    # img_path = os.path.join('HCP_DATA_DIR', '_'.join([sub, IMG_PATH]))
    img_path = os.path.join('..', '_'.join([sub, IMG_PATH]))
    if not os.path.exists(img_path):
        continue
    img = nib.load(img_path)
    X = img.get_fdata()
    X = hcp.normalize(X)
    output_dir = os.path.join(ROOT_DIR, 'npz', sub)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, 'HCP_visual_voxel.npz'),
        V1=X[:,v1_idx],
        V2=X[:,v2_idx],
        V3=X[:,v3_idx],
        V4=X[:,v4_idx]
    )
    print(os.path.join(output_dir, 'HCP_visual_voxel_zj.npz'))