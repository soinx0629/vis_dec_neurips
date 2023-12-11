import sc_mbm.utils as ut
from sc_mbm.mae_for_image import ViTMAELayer
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F

def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

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

class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, num_voxels=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = num_voxels // patch_size
        self.patch_shape = patch_size
        self.num_voxels = num_voxels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels
        # assert V == self.num_voxels, \
        #     f"Input fmri length ({V}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        
        return x


class MAEforFMRIContrast(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, num_voxels=224, patch_size=16, embed_dim=1024, in_chans=1,
                 depth=24, num_heads=16, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, focus_range=None, focus_rate=None, img_recon_weight=1.0, 
                 use_nature_img_loss=False, use_target_as_pos=False, contrastive_loss_weight=0.5, do_self_contrast=False,
                 do_sup_contrast=False, num_sel_neg_contrast=5, do_mask_loss=False,
                 do_distill_loss=False, do_distill_contrast=False, fmri_channels =263, channels = 256, mask_loss_weight=0.5, 
                 distill_loss_weight=0.5, contrast_loss_weight=0.5, do_cross_contrast=False, cross_contrast_loss_weight=0.5, 
                 self_contrast_loss_weight=0.5, negative_mode='unpaired', decoder_freeze=False, distill_contrast_loss_weight=0.1,
                 sep_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(num_voxels, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
  
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True) # encoder to decoder

        # --------------------------------------------------------------------------
        if decoder_freeze:
            for param in self.decoder_embed.parameters():
                param.requires_grad = False
            self.mask_token.requires_grad = False
            # for param in self.decoder_pos_embed.parameters():
            #     param.requires_grad = False
            for param in self.decoder_blocks.parameters():
                param.requires_grad = False
            for param in self.decoder_norm.parameters():
                param.requires_grad = False
            for param in self.decoder_pred.parameters():
                param.requires_grad = False

        # nature image decoder specifics
        if use_nature_img_loss:
            self.nature_img_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.nature_img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.nature_img_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.nature_img_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(2)])

            self.nature_img_decoder_norm = norm_layer(decoder_embed_dim)
            self.nature_img_decoder_pred = nn.Sequential(
                nn.Conv1d(num_patches, 512, kernel_size=1, stride=1, bias=True),
                nn.Linear(decoder_embed_dim, 28*28, bias=True)
            )
            # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.focus_range = focus_range
        self.focus_rate = focus_rate
        self.img_recon_weight = img_recon_weight
        self.use_nature_img_loss = use_nature_img_loss

        self.use_target_as_pos = use_target_as_pos
        self.contrastive_loss_weight = contrastive_loss_weight
        self.do_self_contrast = do_self_contrast
        self.do_sup_contrast = do_sup_contrast
        self.num_sel_neg_contrast = num_sel_neg_contrast
        self.do_mask_loss = do_mask_loss
        self.do_distill_loss = do_distill_loss
        self.mask_loss_weight = mask_loss_weight
        self.distill_loss_weight = distill_loss_weight
        self.contrast_loss_weight = contrast_loss_weight

        self.do_distill_contrast = do_distill_contrast
        self.distill_contrast_loss_weight = distill_contrast_loss_weight

        self.do_cross_contrast = do_cross_contrast
        self.self_contrast_loss_weight= self_contrast_loss_weight
        self.cross_contrast_loss_weight = cross_contrast_loss_weight
        self.negative_mode = negative_mode
        self.sep_loss = sep_loss

        if self.do_distill_loss or self.do_distill_contrast:
            self.distill_loss = nn.KLDivLoss(reduction='batchmean')
            self.distill_loss_mse = nn.MSELoss(reduction='mean')
            self.fmri_channels = fmri_channels
            # self.channels = 256
            # self.fmri_up = FmriTransformerUp(fmri_channels, channels)
            self.conv_up = nn.Conv2d(fmri_channels, 4, (1,1))
            self.linear_up = nn.Linear(4096, 1280)
            self.dis_dropout = nn.Dropout(p=0.1)
            self.dis_dropout_zero = nn.Dropout(p=0.1)

   
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.use_nature_img_loss:
            nature_img_decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.nature_img_decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
            self.nature_img_decoder_pos_embed.data.copy_(torch.from_numpy(nature_img_decoder_pos_embed).float().unsqueeze(0))
            torch.nn.init.normal_(self.nature_img_mask_token, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    
    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        x: (N, L, patch_size)
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_embed.patch_size
        h = x.shape[1]
        
        imgs = x.reshape(shape=(x.shape[0], 1, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if self.focus_range is not None:
            len_mask = L - len_keep
            weights = [1-self.focus_rate] * L
            weights[self.focus_range[0] // self.patch_size : self.focus_range[1] // self.patch_size
                        ] = [self.focus_rate] * (self.focus_range[1] // self.patch_size - self.focus_range[0] // self.patch_size)
            weights = torch.tensor(weights).repeat(N, 1).to(x.device)
            ids_mask = torch.multinomial(weights, len_mask, replacement=False)
            
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if self.focus_range is not None:
            for i in range(N):
                noise[i, ids_mask[i,:]] = 1.1  # set mask portion to 1.1 

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
    
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_nature_img_decoder(self, x, ids_restore):
        # embed tokens
        x = self.nature_img_decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.nature_img_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.nature_img_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.nature_img_decoder_blocks:
            x = blk(x)
        x = self.nature_img_decoder_norm(x)
        # remove cls token
        x = x[:, 1:, :]
        # predictor projection
        # x = x.mean(dim=1, keepdim=True)
        x = self.nature_img_decoder_pred(x)
        x = x.view(x.shape[0], 512, 28, 28)

        return x # n, 512, 28, 28
        
    def forward_nature_img_loss(self, inputs, reconstructions):
        loss = ((torch.tanh(inputs) - torch.tanh(reconstructions))**2).mean()
        if torch.isnan(reconstructions).sum():
            print('nan in reconstructions')
        if torch.isnan(inputs).sum():
            print('nan in inputs')
    
        return loss   

    def info_nce(self, query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        #demo
        #loss = InfoNCE(negative_mode='paired')
        # batch_size, num_negative, embedding_size = 32, 6, 128
        # query = torch.randn(batch_size, embedding_size)
        # positive_key = torch.randn(batch_size, embedding_size)
        # negative_keys = torch.randn(batch_size, num_negative, embedding_size)
        # output = loss(query, positive_key, negative_keys)

        # Check input dimensionality.

        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors
        query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            # logits = query @ transpose(positive_key)
            # print('nxn_cos_sim')
            logits = self.nxn_cos_sim(query, positive_key)

            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(query), device=query.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)
    
    def forward_cross_contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
    
    def forward_contrast_loss(self, pred, pos_sample, neg_sample=None, unpatch=True, temperature=0.1):
        # pred: [N, L, p]
        # pos_sample: [N, L, p]
        # neg_sample: [N, L, p]

        # print(pred.shape, pos_sample.shape, neg_sample.shape)
        if unpatch:
            pred = self.unpatchify(pred).squeeze(1)
        # pos_sample = pos_sample.squeeze(1)
        # neg_sample = neg_sample.squeeze(1)

        if self.negative_mode == 'paired':
            loss = self.info_nce(pred, pos_sample, neg_sample, temperature=temperature, reduction='mean', negative_mode='paired')
        else:
            loss = self.info_nce(pred, pos_sample, temperature=temperature, reduction='mean', negative_mode='unpaired')
        
        return loss
    
    def nxn_cos_sim(self, A, B, dim=1, eps=1e-8):
      numerator = A @ B.T
      A_l2 = torch.mul(A, A).sum(axis=dim)
      B_l2 = torch.mul(B, B).sum(axis=dim)
      denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
      return torch.div(numerator, denominator)
        
    def forward_distill_loss(self, fmri_map, pos_dis_sample, neg_dis_sample=None, temperature=0.5):
        # print('fmri_map', fmri_map.shape, 'pos_dis_sample', pos_dis_sample.shape)
        fmri_map = fmri_map.view([fmri_map.shape[0], -1])
        pos_dis_sample = pos_dis_sample.view([pos_dis_sample.shape[0], -1])
        pos_distill_loss = self.distill_loss(F.log_softmax(fmri_map/temperature,dim=1), F.softmax(pos_dis_sample/temperature, dim=1))
        pos_distill_loss_mse = self.distill_loss_mse(fmri_map, pos_dis_sample)

        return pos_distill_loss + 0.5 * pos_distill_loss_mse
    
    def forward_distill_contrast_loss(self, fmri_map, pos_dis_sample):
        fmri_map = fmri_map.view([fmri_map.shape[0], -1])
        pos_dis_sample = pos_dis_sample.view([pos_dis_sample.shape[0], -1])
        # print('fmri_map', fmri_map.shape, 'pos_dis_sample', pos_dis_sample.shape)
        distill_contrast_loss = self.forward_contrast_loss(fmri_map, pos_dis_sample, unpatch=False,temperature=1)
        return distill_contrast_loss



    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, num_voxels]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, pos_sample=None, neg_sample=None, distill_pos_sample=None, dropout=True, return_fmri_map=False):
        # print('imgs shape', imgs.shape)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p]
        if self.do_distill_loss:
            if dropout:
                fmri_map = self.forward_encoder(self.dis_dropout_zero(imgs), 0)[0]
                # print('fmri map shape 1', fmri_map.shape)
                fmri_map = self.conv_up(fmri_map.view(-1, self.fmri_channels, 32, 32))
                # print('fmri map shape 2', fmri_map.shape)
                fmri_map = self.dis_dropout(fmri_map)

            else:
                fmri_map = self.conv_up(self.forward_encoder(imgs, 0)[0].view(-1, self.fmri_channels, 32, 32))

            fmri_map = self.linear_up(fmri_map.view(fmri_map.shape[0],-1))
            # print('fmri map shape 3', fmri_map.shape,distill_pos_sample.shape)
            distill_loss = self.forward_distill_loss(fmri_map, distill_pos_sample, temperature=0.5)
            # print('distill_loss', distill_loss)
        if self.do_distill_contrast and self.do_distill_loss:
            distill_loss += self.distill_contrast_loss_weight * self.forward_distill_contrast_loss(fmri_map, distill_pos_sample)

        if self.do_distill_contrast and not self.do_distill_loss:
            if dropout:
                fmri_map = self.forward_encoder(self.dis_dropout_zero(imgs), 0)[0]
                # print('fmri map shape 1', fmri_map.shape)
                fmri_map = self.conv_up(fmri_map.view(-1, self.fmri_channels, 32, 32))
                # print('fmri map shape 2', fmri_map.shape)
                fmri_map = self.dis_dropout(fmri_map)
            else:
                fmri_map = self.conv_up(self.forward_encoder(imgs, 0)[0].view(-1, self.fmri_channels, 32, 32))

            fmri_map = self.linear_up(fmri_map.view(fmri_map.shape[0],-1))
            # print('fmri map shape 3', fmri_map.shape)
            distill_loss = self.forward_distill_contrast_loss(fmri_map, distill_pos_sample)
        if not self.do_distill_contrast and not self.do_distill_loss:
            distill_loss = torch.Tensor([0]).to(pred.device)

        if self.do_mask_loss:
            mask_loss = self.forward_loss(imgs, pred, mask)
        else:
            mask_loss = torch.Tensor([0]).to(pred.device)

        if self.do_sup_contrast:
            if self.use_target_as_pos:
                sup_contrast_loss = self.forward_contrast_loss(pred, imgs.squeeze(1), neg_sample)
            else:
                sup_contrast_loss = self.forward_contrast_loss(pred, pos_sample, neg_sample)
        else:
            sup_contrast_loss = torch.Tensor([0]).to(pred.device)

            
        if self.do_self_contrast:
            # if self.use_target_as_pos:
            pred_pos = imgs.squeeze(1)
            pred_neg = []
            if pred.shape[0] > 1:
                if self.negative_mode == 'paired':
                    for ii in range(pred.shape[0]):
                        pred_neg_ii = []
                        for jj in range(pred.shape[0]):
                            if not ii == jj:
                                pred_neg_ii.append(pred[jj])

                        pred_neg.append(self.unpatchify(torch.stack(pred_neg_ii)).squeeze(1))
                                        
                    pred_neg = torch.stack(pred_neg)
                    # print(pred.shape, pred_pos.shape, pred_neg.shape)
                    self_contrast_loss = self.forward_contrast_loss(pred, pred_pos, pred_neg) * self.self_contrast_loss_weight
                elif self.negative_mode == 'unpaired':
                    # print('unpaired pred pred pos shape', pred.shape, pred_pos.shape)
                    self_contrast_loss = self.self_contrast_loss_weight * self.forward_contrast_loss(pred, pred_pos)
            else:
                self_contrast_loss = torch.Tensor([0]).to(pred.device) 
        else:
            self_contrast_loss = torch.Tensor([0]).to(pred.device)

        if self.do_cross_contrast:
            latent_pos, mask_pos, ids_restore_pos = self.forward_encoder(imgs, mask_ratio)
            pred_pos = self.forward_decoder(latent_pos, ids_restore_pos)
            pred_pos = self.unpatchify(pred_pos).squeeze(1)
            # print('latent_pos shape', latent_pos.shape, 'latent_pos shape', pred_pos.shape)
            # cross_contrast_loss = self.forward_cross_contrastive_loss(latent_pos, pred) + self.forward_cross_contrastive_loss(latent_pos)
            cross_contrast_loss = self.cross_contrast_loss_weight * self.forward_contrast_loss(pred, pred_pos) 
            if self.sep_loss and self.do_self_contrast:
                # print('cross pred img shape', pred_pos.shape, imgs.squeeze(1).shape)
                cross_contrast_loss += self.self_contrast_loss_weight * self.forward_contrast_loss(pred_pos, imgs.squeeze(1), unpatch=False)
            
        else:
            cross_contrast_loss = torch.Tensor([0]).to(pred.device)

        contrast_loss = sup_contrast_loss + self_contrast_loss + cross_contrast_loss
        loss = self.mask_loss_weight * mask_loss + self.distill_loss_weight * distill_loss + self.contrastive_loss_weight * contrast_loss

        if not return_fmri_map:
            return loss, pred, mask, mask_loss, contrast_loss, distill_loss
        else:
            return loss, pred, mask, mask_loss, contrast_loss, distill_loss, fmri_map

    # def forward(self, imgs, mask_ratio=0.75, pos_sample=None, neg_sample=None, distill_pos_sample=None, dropout=True):
    #     latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    #     pred = self.forward_decoder(latent, ids_restore)  # [N, L, p]
    #     if self.do_distill_loss:
    #         if dropout:
    #             fmri_map = self.conv_up(self.forward_encoder(self.dis_dropout_zero(imgs), 0)[0].view(-1, self.fmri_channels, 32, 32))
    #             fmri_map = self.dis_dropout(fmri_map)
    #         else:
    #             fmri_map = self.conv_up(self.forward_encoder(imgs, 0)[0].view(-1, self.fmri_channels, 32, 32))

    #         fmri_map = self.linear_up(fmri_map.view(fmri_map.shape[0],-1))
    #         distill_loss = self.forward_distill_loss(fmri_map, distill_pos_sample)
    #         # print('distill_loss', distill_loss)
    #     else:
    #         distill_loss = torch.Tensor([0]).to(pred.device)

    #     if self.do_mask_loss:
    #         mask_loss = self.forward_loss(imgs, pred, mask)
    #     else:
    #         mask_loss = torch.Tensor([0]).to(pred.device)

    #     if self.do_sup_contrast:
    #         if self.use_target_as_pos:
    #             contrast_loss = self.forward_contrast_loss(pred, imgs.squeeze(1), neg_sample)
    #         else:
    #             contrast_loss = self.forward_contrast_loss(pred, pos_sample, neg_sample)
            
    #     elif self.do_self_contrast:
    #         if self.use_target_as_pos:
    #             pred_pos = imgs.squeeze(1)
    #         else:
    #             latent_pos, mask_pos, ids_restore_pos = self.forward_encoder(imgs, 1.2*mask_ratio)
    #             pred_pos = self.forward_decoder(latent_pos, ids_restore_pos)
    #             pred_pos = self.unpatchify(pred_pos).squeeze(1)
    #             # fmri_cos_sim = self.nxn_cos_sim(imgs.squeeze(1), imgs.squeeze(1))
    #             # neg_index = torch.topk(fmri_cos_sim, k=self.num_sel_neg_contrast, dim=1, largest=False, sorted=True).indices

    #             # neg_sample = torch.stack([imgs.squeeze(1)[neg_index_i] for neg_index_i in neg_index])
    #             # loss_contrast = self.forward_contrast_loss(pred, pred_pos, neg_sample)
    #             # pred_neg = self.unpatchify(pred).squeeze(1)
    #             # preg_neg = torch.stack([pred_neg[neg_index_i] for neg_index_i in neg_index])

    #             # print(pred.shape, pred_pos.shape, neg_sample.shape, neg_index.shape, fmri_cos_sim.shape, preg_neg.shape)
                
    #         pred_neg = []
    #         if pred.shape[0] > 1:
    #             for ii in range(pred.shape[0]):
    #                 pred_neg_ii = []
    #                 for jj in range(pred.shape[0]):
    #                     if not ii == jj:
    #                         pred_neg_ii.append(pred[jj])

    #                 pred_neg.append(self.unpatchify(torch.stack(pred_neg_ii)).squeeze(1))
                                    
    #             pred_neg = torch.stack(pred_neg)
    #             # print(pred.shape, pred_pos.shape, pred_neg.shape)
    #             contrast_loss = self.forward_contrast_loss(pred, pred_pos, pred_neg)
    #         else:
    #             contrast_loss = torch.Tensor([0.]).to(pred.device)
        
    #     else:
    #         contrast_loss = torch.Tensor([0]).to(pred.device)
        
    #     loss = self.mask_loss_weight * mask_loss + self.distill_loss_weight * distill_loss + self.contrast_loss_weight * contrast_loss

    #     return loss, pred, mask, mask_loss, contrast_loss, distill_loss

class MAEforFMRICross(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, num_voxels=224, patch_size=16, embed_dim=1024, in_chans=1,
                 depth=24, num_heads=16, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, focus_range=None, focus_rate=None, img_recon_weight=1.0, 
                 use_nature_img_loss=False, do_cross_attention=False, cross_encoder_config=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(num_voxels, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
  
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        
        self.do_cross_attention = do_cross_attention
        if self.do_cross_attention:
            self.cross_blocks = nn.ModuleList([ViTMAELayer(cross_encoder_config, True) for _ in range(cross_encoder_config.num_cross_encoder_layers)])
        self.do_cross_residual = cross_encoder_config.do_cross_residual
        #to be removed, temporary computation
        self.cross_map_in = nn.Linear(embed_dim, 768, bias=True)
        self.cross_map_out = nn.Linear(768, embed_dim, bias=True)

        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True) # encoder to decoder
        # --------------------------------------------------------------------------

        # nature image decoder specifics
        if use_nature_img_loss:
            self.nature_img_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.nature_img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.nature_img_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.nature_img_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(2)])

            self.nature_img_decoder_norm = norm_layer(decoder_embed_dim)
            self.nature_img_decoder_pred = nn.Sequential(
                nn.Conv1d(num_patches, 512, kernel_size=1, stride=1, bias=True),
                nn.Linear(decoder_embed_dim, 28*28, bias=True)
            )
            # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.focus_range = focus_range
        self.focus_rate = focus_rate
        self.img_recon_weight = img_recon_weight
        self.use_nature_img_loss = use_nature_img_loss
   
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.use_nature_img_loss:
            nature_img_decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.nature_img_decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
            self.nature_img_decoder_pos_embed.data.copy_(torch.from_numpy(nature_img_decoder_pos_embed).float().unsqueeze(0))
            torch.nn.init.normal_(self.nature_img_mask_token, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    
    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        x: (N, L, patch_size)
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_embed.patch_size
        h = x.shape[1]
        
        imgs = x.reshape(shape=(x.shape[0], 1, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if self.focus_range is not None:
            len_mask = L - len_keep
            weights = [1-self.focus_rate] * L
            weights[self.focus_range[0] // self.patch_size : self.focus_range[1] // self.patch_size
                ] = [self.focus_rate] * (self.focus_range[1] // self.patch_size - self.focus_range[0] // self.patch_size)
            weights = torch.tensor(weights).repeat(N, 1).to(x.device)
            ids_mask = torch.multinomial(weights, len_mask, replacement=False)
            
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if self.focus_range is not None:
            for i in range(N):
                noise[i, ids_mask[i,:]] = 1.1  # set mask portion to 1.1 

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, img_support=None):
        # embed patches
        x = self.patch_embed(x)
    
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.do_cross_attention==True and img_support is not None:
            x = self.cross_map_in(x)
            cross_x = x.clone()
            # print('cross_x', cross_x.shape)
            for blk in self.cross_blocks:
                cross_x_full = blk(cross_x, hidden_states_mod2=img_support)
                cross_x = cross_x_full[0]

            if self.do_cross_residual:
                x = x + cross_x
            else:
                x = cross_x
            x = self.cross_map_out(x)

        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # if image_support is not None:
        #     x = torch.cat([x, image_support], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
        
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, num_voxels]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches
        return loss

    def forward(self, imgs, valid_idx=None, mask_ratio=0.75, image_support=None, encoder_only=False):
        if self.do_cross_attention:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, img_support=image_support)
        else:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        if encoder_only:
            return latent
        else:
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p]
            loss = self.forward_loss(imgs, pred, mask)

            return loss, pred, mask



class fmri_encoder(nn.Module):
    def __init__(self, num_voxels=224, patch_size=16, embed_dim=1024, in_chans=1,
                 depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, global_pool=True):
        super().__init__()
        self.patch_embed = PatchEmbed1D(num_voxels, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)
        x = self.norm(x)

        return x  

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)  # N, n_seq, embed_dim
        latent = self.forward_encoder(imgs) # N, n_seq, embed_dim
        return latent # N, n_seq, embed_dim
    
    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 
    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output