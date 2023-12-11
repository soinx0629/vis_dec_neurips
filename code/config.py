import os
import numpy as np


class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass 
class Config_MBM_finetune_contrast: # back compatibility
    pass 

class Config_MBM_fMRI(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 100
        self.clip_grad = 0.8
        
        # Model Parameters
        self.mask_ratio = 0.75
        self.patch_size = 16
        self.embed_dim = 1024 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0

        # Project setting
        self.root_path = '.'
        self.output_path = self.root_path
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0

class Config_MBM_fMRI_contrast(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 100
        self.clip_grad = 0.8
        
        # Model Parameters
        self.mask_ratio = 0.75
        self.patch_size = 16
        self.embed_dim = 1024 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0

        # Project setting
        self.root_path = '.'
        self.output_path = self.root_path
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0

        self.do_self_contrast = False
        self.do_sup_contrast = False
        self.do_mask_loss = False
        self.do_early_stop = False
        self.mask_loss_weight = 1
        self.contrast_loss_weight = 0.5
        self.use_target_as_pos = False

        self.do_distill_loss = False
        self.do_distill_contrast = False
        self.distill_loss_weight = 0.1
        self.contrast_loss_weight = 0.5

        self.do_early_stop = False
        self.finetune_on_train_contrast = True

        self.do_cross_contrast = False
        self.cross_contrast_loss_weight = 0.5
        self.negative_mode = 'unpaired'
        self.fmri_channels = 292


class Config_MBM_finetune(Config_MBM_finetune):
    def __init__(self):
        
        # Project setting
        self.root_path = '.'
        self.output_path = self.root_path
        self.kam_path = os.path.join(self.root_path, 'data/Kamitani/npz')
        self.bold5000_path = os.path.join(self.root_path, 'data/BOLD5000')
        self.dataset = 'GOD' # GOD  or BOLD5000
        self.pretrain_mbm_path = os.path.join(self.root_path, f'pretrains/{self.dataset}/fmri_encoder.pth') 

        self.include_nonavg_test = True
        self.kam_subs = ['sbj_3']
        self.bold5000_subs = ['CSI4']

        # Training Parameters
        self.lr = 5.3e-5
        self.weight_decay = 0.05
        self.num_epoch = 15
        self.batch_size = 16 if self.dataset == 'GOD' else 4 
        self.mask_ratio = 0.75 
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.warmup_epochs = 2
        self.min_lr = 0.
        self.use_nature_img_loss = False

        # distributed training
        self.local_rank = 0

class Config_MBM_finetune_cross(Config_MBM_finetune):
    def __init__(self):
        self.load_pretrain_state = 1
        
        # Project setting
        self.root_path = '.'
        self.output_path = self.root_path
        self.kam_path = os.path.join(self.root_path, 'data/Kamitani/npz')
        self.bold5000_path = os.path.join(self.root_path, 'data/BOLD5000')
        self.dataset = 'GOD' # GOD  or BOLD5000
        self.pretrain_mbm_path = os.path.join(self.root_path, f'pretrains/{self.dataset}/fmri_encoder.pth') 

        self.include_nonavg_test = False
        self.kam_subs = 'sbj_1'
        self.bold5000_subs = 'CSI1'

        # Training Parameters
        self.lr = 5.3e-5
        self.weight_decay = 0.05
        self.num_epoch = 15
        self.batch_size = 16 if self.dataset == 'GOD' else 4 
        self.mask_ratio = 0.75 
        self.img_mask_ratio = 0.75 
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.warmup_epochs = 2
        self.min_lr = 0.
        self.use_nature_img_loss = False

        # distributed training
        self.local_rank = 0
        self.vit_mae_model = "facebook/vit-mae-base"
        self.num_cross_encoder_layers = 3
        self.do_cross_attention = True
        self.do_cross_residual = True
        self.img_decoder_layers = 6
        self.fmri_decoder_layers = 6

        self.bold5000_train_subs = ['CSI1', 'CSI2', 'CSI3']
        self.bold5000_test_subs = ['CSI4']
        self.kam_train_subs = ['sbj_1', 'sbj_2', 'sbj_3', 'sbj_4']
        self.kam_test_subs = ['sbj_5']
        self.target_sub_train_proportion = 1
        self.fmri_recon_weight=1.0
        self.img_recon_weight=1.0
        self.cross_sub = False
        if self.dataset == 'GOD':
            self.wandb_name = f"cross_att_{self.dataset}_{self.kam_subs[0]}_fmriw{self.fmri_recon_weight}_imgw{self.img_recon_weight}_fmar{self.mask_ratio}_imar{self.img_mask_ratio}"
        else:
            self.wandb_name = f"cross_att_{self.dataset}_{self.bold5000_subs[0]}_fmriw{self.fmri_recon_weight}_imgw{self.img_recon_weight}_fmar{self.mask_ratio}_imar{self.img_mask_ratio}"

class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '.'
        self.kam_path = os.path.join(self.root_path, 'data/Kamitani/npz')
        self.bold5000_path = os.path.join(self.root_path, 'data/BOLD5000')
        self.roi = 'VC'
        self.patch_size = 16

        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/semantic')
        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/label2img')
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/text2img-large')
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/layout2img')
        
        self.dataset = 'GOD' # GOD or BOLD5000
        self.kam_subs = ['sbj_3']
        self.bold5000_subs = ['CSI4']
        self.pretrain_mbm_path = os.path.join(self.root_path, f'pretrains/{self.dataset}/fmri_encoder.pth') 

        self.img_size = 256

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 500
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None # os.path.join(self.root_path, 'results/generation/25-08-2022-08:02:55/checkpoint.pth')
        

def merge_needed_cross_config(config_pretrain, config_cross, config_img_mae, additonal_config=None):
    all_cross_config = {}
    all_cross_config['patch_size'] = config_pretrain.patch_size
    all_cross_config['embed_dim'] = config_pretrain.embed_dim
    all_cross_config['decoder_embed_dim'] = config_pretrain.decoder_embed_dim
    all_cross_config['depth'] = config_pretrain.depth
    all_cross_config['num_heads'] = config_pretrain.num_heads
    all_cross_config['decoder_num_heads'] = config_pretrain.decoder_num_heads
    all_cross_config['mlp_ratio'] = config_pretrain.mlp_ratio
    all_cross_config['decoder_depth'] = config_cross.fmri_decoder_layers
    all_cross_config['do_cross_attention'] = config_cross.do_cross_attention
    all_cross_config['target_sub_train_proportion'] = config_cross.target_sub_train_proportion
    all_cross_config['bold5000_train_subs'] = config_cross.bold5000_train_subs
    all_cross_config['bold5000_test_subs'] = config_cross.bold5000_test_subs
    all_cross_config['kam_train_subs'] = config_cross.kam_train_subs
    all_cross_config['kam_test_subs'] = config_cross.kam_test_subs
    all_cross_config['target_sub_train_proportion'] = config_cross.target_sub_train_proportion
    all_cross_config['fmri_recon_weight'] = config_cross.fmri_recon_weight
    all_cross_config['img_recon_weight'] = config_cross.img_recon_weight
    all_cross_config['cross_img_encoder_config'] = config_img_mae
    
    if additonal_config is not None:
        all_cross_config.update(additonal_config)

    return all_cross_config

def merge_needed_vqgan_config(config_pretrain, config_vqgan, additonal_config=None):
    all_cross_config = {}
    all_cross_config['patch_size'] = config_pretrain.patch_size
    all_cross_config['embed_dim'] = config_pretrain.embed_dim
    all_cross_config['decoder_embed_dim'] = config_pretrain.decoder_embed_dim
    all_cross_config['depth'] = config_pretrain.depth
    all_cross_config['num_heads'] = config_pretrain.num_heads
    all_cross_config['decoder_num_heads'] = config_pretrain.decoder_num_heads
    all_cross_config['mlp_ratio'] = config_pretrain.mlp_ratio
    all_cross_config['use_vqgan_recon_loss'] = config_vqgan.use_vqgan_recon_loss
    all_cross_config['do_distill'] = config_vqgan.do_distill
    all_cross_config['freeze_vqgan_dec'] = config_vqgan.freeze_vqgan_dec
    all_cross_config['freeze_vqgan_enc'] = config_vqgan.freeze_vqgan_dec
    all_cross_config['dropout_rate'] = config_vqgan.dropout_rate
    all_cross_config['gaussian_noise_weight'] = config_vqgan.gaussian_noise_weight
    all_cross_config['vqgan_model_path'] = config_vqgan.vqgan_model_path

    if additonal_config is not None:
        all_cross_config.update(additonal_config)

    return all_cross_config
