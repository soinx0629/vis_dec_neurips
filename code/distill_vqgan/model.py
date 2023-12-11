import pytorch_lightning as pl
import torch  
from distill_vqgan.modules import FmriVQAEModel
from distill_vqgan.losses.vqperceptual import VQLPIPSWithDiscriminator
from code.dc_ldm.modules.ema import LitEma
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR
from packaging import version

class DistillationFmriVQGAN(pl.LightningModule):

    def __init__(self,
                 fmri_vq_model_ckpt_path,
                 fmri_vq_model_config,
                 loss_config,
                 fmri_vit_ckpt_path,
                 use_ema = True,
                 freeze_fmri_vit=True,
                 fmri_channels =263, # output of fmri vit dimension
                 channels = 256,
                 lr_g_factor = 1,
                 ckpt_path = None): # alwarys 256):
        super().__init__()
        self.model = FmriVQAEModel(fmri_vq_model_ckpt_path,
                              fmri_vq_model_config,
                              fmri_vit_ckpt_path,
                              freeze_fmri_vit=True,
                              fmri_channels =263, 
                              channels = 256)
        self.learning_rate = None 
        self.lr_g_factor = lr_g_factor
        self.loss = VQLPIPSWithDiscriminator(**loss_config)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAS of  {len(list(self.model_ema.buffers()))}.")
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path,ignore_keys=[]):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restoring from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

        
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


    def forward(self, x, img):
        tea_img_feat, std_fmri_feat, xrec, emb_loss, info = self.model(x, img)
        return tea_img_feat, std_fmri_feat, xrec, emb_loss, info
    
    def get_input(self,batch):
        fmri = batch['fmri']
        img = batch['image'].permute(0,3,1,2).float()
        return fmri, img 
    

    def get_last_layer(self):
        return self.model.fmri_vq_student_model.decoder.conv_out.weight 


    def training_step(self, batch, batch_idx, optimizer_idx):
        fmri, x = self.get_input(batch)
        tea_img_feat, std_fmri_feat, xrec, qloss, info = self(fmri, x)
        if optimizer_idx == 0:
            # autoencoder and distallation
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx,self.global_step,
                      tea_img_feat, std_fmri_feat, last_layer=self.get_last_layer(), split="train",)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                      tea_img_feat, std_fmri_feat, last_layer=self.get_last_layer(),split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            return discloss 

            
    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict 

    def _validation_step(self, batch, batch_idx, suffix=""):
        fmri, x = self.get_input(batch)
        tea_img_feat, std_fmri_feat, xrec, qloss, info = self(fmri, x)
        
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,self.global_step,
                      tea_img_feat, std_fmri_feat, last_layer=self.get_last_layer(), split="val"+suffix,)
        
        
        discloss, log_dict_disc = self.loss(qloss, x, xrec,1, self.global_step,
                      tea_img_feat, std_fmri_feat, last_layer=self.get_last_layer(),split="val"+suffix)

        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        dis_loss = log_dict_ae[f"val{suffix}/distillation_loss"]
 
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/distillation_loss", dis_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
            del log_dict_ae[f"val{suffix}/distillation_loss"]

        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)


    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.model.fmri_vq_student_model.encoder.parameters())+
                                  list(self.model.fmri_vq_student_model.decoder.parameters())+
                                  list(self.model.fmri_vq_student_model.quantize.parameters())+
                                  list(self.model.fmri_vq_student_model.quant_conv.parameters())+
                                  list(self.model.fmri_vq_student_model.post_quant_conv.parameters())+
                                  list(self.model.fmri_up.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        return [opt_ae, opt_disc], []
    
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
        

    



