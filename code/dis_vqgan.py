from dc_ldm.models.autoencoder import VQModelInterface, VQModel
from distill_vqgan.modules import load_vqmodel
from distill_vqgan.losses.vqperceptual import VQLPIPSWithDiscriminator
from distill_vqgan.configs import vqmodel_config, distilla_fmri_loss_config
import torch
import torch.nn as nn 
from contextlib import contextmanager


class VQGan(VQModel):
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
    
    def init_from_ckpt(self, path, ignore_keys=list()):
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


class FmriVQGan(nn.Module):

    def __init__(self,
                 vq_ckpt,
                 loss_config=distilla_fmri_loss_config,
                 vq_model_config=vqmodel_config,
                ):
        ### Load Fmri model
        #定义 fmri encoder
        super().__init__()
        self.vqmodel = VQGan(vq_model_config)
        if vq_ckpt is not None:
            self.vqmodel.init_from_ckpt(vq_ckpt)
        self.loss = VQLPIPSWithDiscriminator(**loss_config)
       
        #self.vqmodel.freeze()
    
    def get_input(self, batch):
        # return {
        #     "fmri_feat": batch["fmri_feat"],
        #     "image": batch["image"]
        # }
        return batch["fmri_feat"], batch["image"]
    
    def get_last_layer(self):
        return self.vqmodel.decoder.conv_out.weight 

    def forward_in(self, fmri_feat, image, return_enc_feat=False):
        image_encoder_feat = self.vqmodel.encode_to_prequant(image)
        # fmri_feat = None #通过定义的fmri encoder得到fmri feature 64 64 3
        # print("fmri_feat", fmri_feat.shape)
        # print("image_encoder_feat", image_encoder_feat.shape)
        image_feat = image_encoder_feat + fmri_feat
        image, emb_loss, info = self.vqmodel.decode_from_prequant(image_feat)
        if not return_enc_feat:
            return image, emb_loss, info
        else:
            return image, emb_loss, info, image_encoder_feat
    
    def forward(self, batch, optimizer_idx=None, global_step=None, return_enc_feat=False):
        fmri, x = self.get_input(batch)
        if return_enc_feat:
            enc_feat = self.vqmodel.encode_to_prequant(x)
            return enc_feat
        else:
            xrec, q_loss, _ =  self.forward_in(fmri,x)

            if optimizer_idx == 0:
                aeloss, _ = self.loss(q_loss, x, xrec, optimizer_idx, global_step,
                                last_layer=self.get_last_layer(), split="train",)
                # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)

                return aeloss

            if optimizer_idx == 1:
                discloss, _ = self.loss(q_loss, x, xrec, optimizer_idx, global_step,
                        last_layer=self.get_last_layer(),split="train")
                # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
                return discloss

    
    # def validataion_step(self, batch, batch_idx):
    #     pass 
    

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    # def configure_optimizers(self):
    #     lr_d = self.learning_rate
    #     lr_g = self.lr_g_factor*self.learning_rate
    #     print("lr_d", lr_d)
    #     print("lr_g", lr_g)
    #     # 需要更新fmri encoder的参数 
    #     opt_ae = torch.optim.Adam(self.fmri_transformer.parameters(),
    #                               lr=lr_g, betas=(0.5, 0.9))
    #     opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
    #                                 lr=lr_d, betas=(0.5, 0.9))

    #     return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        fmri, x = self.get_input(batch)
        x = x.to(self.device)
        _, _, xrec, _, _ = self(fmri, x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self,x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
      


