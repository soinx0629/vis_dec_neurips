import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib.pyplot as plt
import wandb
import copy
from transformers import AutoFeatureExtractor
import shutil

# own code
from config import Config_MBM_finetune_cross, merge_needed_cross_config
from dataset import create_Kamitani_dataset_distill, create_BOLD5000_dataset_classify
from sc_mbm.mae_for_fmri import MAEforFMRICross
from sc_mbm.mae_for_image import ViTMAEForPreTraining, ViTMAEConfig
from sc_mbm.trainer import train_one_epoch_cross, eval_one_epoch_cross
from sc_mbm.trainer import NativeScalerWithGradNormCount as NativeScaler
from sc_mbm.utils import save_model_merge_conf

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."

class wandb_logger:
    def __init__(self, config):
        wandb.init( project='vis-dec',
                    group="stepA_sc-mbm_crossatt_singlesub",
                    anonymous="allow",
                    config=config,
                    reinit=True,
                    name=config.wandb_name)

        self.config = config
        self.step = None
    
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE finetuning on Test fMRI', add_help=False)

    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--mask_ratio', type=float)
    parser.add_argument('--img_mask_ratio', type=float)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--include_nonavg_test', type=bool)   
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--img_decoder_layers', type=int)
    parser.add_argument('--fmri_decoder_layers', type=int)
    parser.add_argument('--fmri_recon_weight', type=float)
    parser.add_argument('--img_recon_weight', type=float)
    parser.add_argument('--kam_subs', type=str)
    parser.add_argument('--bold5000_subs', type=str)
    parser.add_argument('--load_pretrain_state', type=int)
     
                        
    return parser

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def add_weight_decay(models, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def load_model_image(config):
    image_feature_extractor = AutoFeatureExtractor.from_pretrained(config.vit_mae_model)
    model_image_config =  ViTMAEConfig.from_pretrained(config.vit_mae_model)
    model_image = ViTMAEForPreTraining.from_pretrained(config.vit_mae_model)

    model_image_config.num_cross_encoder_layers = config.num_cross_encoder_layers
    # model_image_config.do_cross_attention = config.do_cross_attention
    model_image_config.do_cross_residual = config.do_cross_residual
    model_image_config.decoder_num_hidden_layers = config.img_decoder_layers
    model_image_new = ViTMAEForPreTraining(model_image_config, do_cross_attention=config.do_cross_attention)

    pretrained_state_dict = model_image.state_dict()
    new_model_state_dict = model_image_new.state_dict()
    for key in pretrained_state_dict.keys():
        if key in new_model_state_dict:
            new_model_state_dict[key] = pretrained_state_dict[key]

    model_image_new.load_state_dict(new_model_state_dict)

    # model_image_new.vit.eval()
    

    for param in model_image_new.vit.parameters():
        param.requires_grad = False

    for param in model_image_new.vit.encoder.cross_map_in.parameters():
        param.requires_grad = True

    for param in model_image_new.vit.encoder.layer_cross.parameters():
        param.requires_grad = True

    model_image_new.decoder.train()
    
    for param in model_image_new.decoder.parameters():
        param.requires_grad = True

    return model_image_new, model_image_config, image_feature_extractor

def main(config):

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')
    sd = torch.load(config.pretrain_mbm_path, map_location='cpu')
    config_pretrain = sd['config']
    
    
    output_sub = config.bold5000_subs if config.dataset == 'BOLD5000' else config.kam_subs
    output_path = os.path.join(config.output_path, 'results', 'fmri_finetune_{}_{}'.format(config.dataset, output_sub),  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))

    config.output_path = output_path
    if config.dataset == 'GOD':
        config.wandb_name = f"cross_att_{config.dataset}_{config.kam_subs}_fmriw{config.fmri_recon_weight}_imgw{config.img_recon_weight}_fmar{config.mask_ratio}_imar{config.img_mask_ratio}_fmridl{config.fmri_decoder_layers}_imgdl{config.img_decoder_layers}_pretr{config.load_pretrain_state}_with_{config.pretrain_mbm_path.split('/')[-1]}"
    else:
        config.wandb_name = f"cross_att_{config.dataset}_{config.bold5000_subs}_fmriw{config.fmri_recon_weight}_imgw{config.img_recon_weight}_fmar{config.mask_ratio}_imar{config.img_mask_ratio}_fmridl{config.fmri_decoder_layers}_imgdl{config.img_decoder_layers}_pretr{config.load_pretrain_state}_with_{config.pretrain_mbm_path.split('/')[-1]}"
    logger = wandb_logger(config) if config.local_rank == 0 else None
    
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config_pretrain.seed)
    np.random.seed(config_pretrain.seed)

    # create model

    model_image, model_image_config, image_feature_extractor = load_model_image(config)
    

    model_image.to(device)
    # model_image_without_ddp = model_image

    num_voxels = (sd['model']['pos_embed'].shape[1] - 1)* config_pretrain.patch_size
    model = MAEforFMRICross(num_voxels=num_voxels, patch_size=config_pretrain.patch_size, embed_dim=config_pretrain.embed_dim,
                    decoder_embed_dim=config_pretrain.decoder_embed_dim, depth=config_pretrain.depth, 
                    num_heads=config_pretrain.num_heads, decoder_num_heads=config_pretrain.decoder_num_heads, 
                    mlp_ratio=config_pretrain.mlp_ratio, focus_range=None, use_nature_img_loss=False, 
                    do_cross_attention=config.do_cross_attention, cross_encoder_config=model_image_config,
                    decoder_depth=config.fmri_decoder_layers) 
    if config.load_pretrain_state == 1:
        model.load_state_dict(sd['model'], strict=False)

    model.to(device)
    model_without_ddp = model

    
    # create dataset and dataloader
    if config.dataset == 'GOD':
        train_set, test_set = create_Kamitani_dataset_distill(path=config.kam_path, patch_size=config_pretrain.patch_size, 
                                subjects=config.kam_subs, fmri_transform=torch.FloatTensor, include_nonavg_test=config.include_nonavg_test,
                                return_image_name=True)
    elif config.dataset == 'BOLD5000':
        train_set, test_set = create_BOLD5000_dataset_classify(path=config.bold5000_path, patch_size=config_pretrain.patch_size, 
                fmri_transform=torch.FloatTensor, subjects=config.bold5000_subs, include_nonavg_test=config.include_nonavg_test)
    else:
        raise NotImplementedError
    
    if train_set.fmri.shape[-1] < num_voxels:
        train_set.fmri = np.pad(train_set.fmri, ((0,0), (0, num_voxels - train_set.fmri.shape[-1])), 'wrap')
    else:
        train_set.fmri = train_set.fmri[:, :num_voxels]

    # print(test_set.fmri.shape)
    if test_set.fmri.shape[-1] < num_voxels:
        test_set.fmri = np.pad(test_set.fmri, ((0,0), (0, num_voxels - test_set.fmri.shape[-1])), 'wrap')
    else:
        test_set.fmri = test_set.fmri[:, :num_voxels]

    print(f'Dataset size: {len(train_set)}, {len(test_set)}')
    sampler = torch.utils.data.DistributedSampler(train_set) if torch.cuda.device_count() > 1 else torch.utils.data.RandomSampler(train_set) 
    dataloader_hcp = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler)
    test_sampler = torch.utils.data.DistributedSampler(test_set) if torch.cuda.device_count() > 1 else torch.utils.data.RandomSampler(test_set) 
    dataloader_hcp_test = DataLoader(test_set, batch_size=config.batch_size, sampler=test_sampler)

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=config.use_nature_img_loss)

        model_image = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_image)
        model_image = DistributedDataParallel(model_image, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=config.use_nature_img_loss)

    param_groups = add_weight_decay([model, model_image], config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    

    print(optimizer)
    loss_scaler = NativeScaler()
    # loss_scaler_img = NativeScaler()

    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)

    cor_list = []
    eval_cor_list = []
    eval_cor_init = 0.5
    best_eval_corr_epoch = 0
    saved_epoch_list = []
    start_time = time.time()
    print('Finetuning MAE on train fMRI ... ...')
    addition_config = {'num_voxels':num_voxels}
    merged_config = merge_needed_cross_config(config_pretrain, config, model_image_config, addition_config) 
    for ep in range(config.num_epoch):
        ckpt_file_name = f'checkpoint_singlesub_{config.wandb_name}_epo{ep}_mergconf.pth'
        if torch.cuda.device_count() > 1: 
            sampler.set_epoch(ep) # to shuffle the data at every epoch
        cor = train_one_epoch_cross(model, model_image, dataloader_hcp, optimizer, device, ep, loss_scaler, logger, config, start_time, model_without_ddp,
                                    img_feature_extractor=image_feature_extractor, fmri_recon_weight=config.fmri_recon_weight, 
                                    img_recon_weight=config.img_recon_weight)
        eval_cor = eval_one_epoch_cross(model, model_image, dataloader_hcp_test, device, ep, logger, config, start_time, model_without_ddp,  
                                        img_feature_extractor=image_feature_extractor) 
        cor_list.append(cor)
        eval_cor_list.append(eval_cor)


        if ep != 0 and config.local_rank == 0:
            # plot_recon_figures(model, device, test_set, output_path, 5, config, logger, model_without_ddp)
            if eval_cor > eval_cor_init:
                save_model_merge_conf(config_pretrain, ep, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path, f'checkpoints_{ep}'), merged_config, ckpt_file_name)
                eval_loss_increase_count = 0 
                eval_cor_init = eval_cor
                best_eval_corr_epoch = ep
                saved_epoch_list.append(ep)
                if len(saved_epoch_list) > 2:
                    for del_ep in saved_epoch_list[:-2]:
                        print('Deleting model at ep {}'.format(del_ep))
                        shutil.rmtree(os.path.join(output_path, f'checkpoints_{del_ep}'))
                    saved_epoch_list = saved_epoch_list[-2:]
                print('Saving model at ep {} eval_image_corr_train {} eval_image_corr_test {}, best ep {},  best eval_loss {}'.format(
                    ep, cor, eval_cor, best_eval_corr_epoch, eval_cor_init))
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if logger is not None:
        logger.log('max cor', np.max(cor_list), step=config.num_epoch-1)
        logger.log('max eval cor', np.max(eval_cor_list), step=config.num_epoch-1)
        logger.finish()
    return



@torch.no_grad()
def plot_recon_figures(model, device, dataset, output_path, num_figures = 5, config=None, logger=None, model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Masked Ground-truth')
    axs[0,2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))['fmri']
        sample = sample.to(device)
        _, pred, mask = model(sample, mask_ratio=config.mask_ratio)
        sample_with_mask = model_without_ddp.patchify(sample).to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
        pred = model_without_ddp.unpatchify(pred).to('cpu').numpy().reshape(-1)
        sample = sample.to('cpu').numpy().reshape(-1)
        mask = mask.to('cpu').numpy().reshape(-1)
        # cal the cor
        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
        s = 0
        for x, m in zip(sample_with_mask,mask):
            if m == 0:
                ax[1].plot(x_axis[s:s+len(x)], x, color='#1f77b4')
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_MBM_finetune_cross()
    config = update_config(args, config)
    main(config)
    
