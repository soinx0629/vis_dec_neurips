import torch 
import torch.nn as nn 
import torch.nn.functional as F
from dc_ldm.models.autoencoder import VQModelInterface, VQModel
from sc_mbm.mae_for_fmri import MAEforFMRI
from distill_vqgan.configs import vqmodel_config

def load_vqmodel(configs, ckpt_path=None):

    #sd = torch.load(ckpt_path, map_location="cpu")
    vq_model = FmriVQModel(configs)
    if ckpt_path is not None:
        vq_model._init__from_ckpt(ckpt_path)
    return vq_model 
    
    
def load_fmri_transformer_encdoer(ckpt_path=None):
    sd = torch.load(ckpt_path, map_location="cpu")
    config = sd['config']
   
    model = FmriTransformerEncoder(
        num_voxels=config.num_voxels ,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=1,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        norm_layer=nn.LayerNorm,
        focus_range=None,
        focus_rate=None,
        img_recon_weight=1.0,
        use_nature_img_loss=False,
    )
    if ckpt_path is not None:
        model._init__from_ckpt(ckpt_path)
    return model 

def normalization(channels):
    return GroupNorm32(32, channels)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, (3,3),1,1) # keep dim

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x



class UpResBlock(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, (3,3), 1, 1)
        )
        self.h_up = Upsample(channels, False)
        self.x_up = Upsample(channels, False)
        if channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels,out_channels,(3,3),1,1)

    def forward(self,x):
        in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
        h = in_rest(x)
        h = self.h_up(h)
        x = self.x_up(x)
        h = in_conv(h)
        return self.skip_connection(x) + h 


class FmriVQModel(VQModel):
    def __init__(self, kwargs):
        super().__init__(**kwargs)
    
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad=False
        print(f"Freeze {self.__class__.__name__}") 
    
    def unfreeze(self):
        self.train()
        for param in self.parameters():
            param.requires_grad=True 
        print(f"Unfreeze {self.__class__.__name__}")
    
    def _init__from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")['state_dict']
        keys = list(sd.keys())
        vq_sd = {}
        for k in keys:
            if "first_stage_model" in k:
                new_k = k.replace("first_stage_model.","")
                vq_sd[new_k] = sd[k]
        
        keys = list(vq_sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del vq_sd[k]
        missing, unexpected = self.load_state_dict(vq_sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
    


class FmriTransformerUp(nn.Module):

    def __init__(self, in_channels=263, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, (1,1)) 
        self.upblock1 = UpResBlock(256,128)
        self.upblock2 = UpResBlock(128,64)
        self.upblock3 = UpResBlock(64, 32)
        self.last_conv = nn.Conv2d(32,3,(3,3),1,1)

    def forward(self, x):
         x = self.conv1(x)
         x_up1 = self.upblock1(x)
         x_up2 = self.upblock2(x_up1)
         x_up3 = self.upblock3(x_up2)
         x_out = self.last_conv(x_up3)
         return x_out 


class FmriTransformerEncoder(MAEforFMRI):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad=False
        print(f"Freeze {self.__class__.__name__}") 
    
    def unfreeze(self):
        self.train()
        for param in self.parameters():
            param.requires_grad=True 
        print(f"Unfreeze {self.__class__.__name__}")
    
    def _init__from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")['model']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
            

class FmriVQAEModel(nn.Module):
    def __init__(self,
                 fmri_vq_model_ckpt_path,
                 fmri_vq_model_config,
                 fmri_vit_ckpt_path,
                 freeze_fmri_vit=True,
                 fmri_channels =263, # output of fmri vit dimension
                 channels = 256 , # alwarys 256
                 ):
        super().__init__()

        self.fmri_channels = fmri_channels
        self.fmri_vit = load_fmri_transformer_encdoer(fmri_vit_ckpt_path)
        if freeze_fmri_vit:
            self.fmri_vit.freeze()
        else:
            self.fmri_vit.unfreeze()
        
        self.fmri_up = FmriTransformerUp(fmri_channels, channels)
        self.fmri_vq_student_model = load_vqmodel(fmri_vq_model_config, None)
        self.fmri_vq_teacher_model = load_vqmodel(fmri_vq_model_config, fmri_vq_model_ckpt_path)
        self.fmri_vq_teacher_model.freeze()


    def forward(self, x, img):
        fmri_feat = self.fmri_vit.forward_encoder(x,0)[0].view(-1, self.fmri_channels, 32, 32)
        fmri_map = self.fmri_up(fmri_feat)
        tea_img_feat = self.teacher_forward(img)
       
        std_fmri_feat, xrec, emb_loss, info = self.student_forward(fmri_map)
        return tea_img_feat, std_fmri_feat, xrec, emb_loss, info
    

    def student_forward(self, fmri_feat):
        fmri_feat = self.fmri_vq_student_model.encode_to_prequant(fmri_feat)
        xrec, emb_loss, info = self.fmri_vq_student_model.decode_from_prequant(fmri_feat)
        return fmri_feat, xrec, emb_loss, info


    def teacher_forward(self, img):
        img_feat = self.fmri_vq_teacher_model.encode_to_prequant(img)
        return img_feat 



if __name__ == "__main__":
    # fmri_encoder = load_fmri_transformer_encdoer(
    #     ckpt_path="pretrains/BOLD5000/fmri_encoder.pth"
    #     )
    
    vqmodel_ckpt = "pretrains/ldm/label2img/model.ckpt"
    sd = torch.load(vqmodel_ckpt, map_location="cpu")
    
    import ipdb 
    ipdb.set_trace()