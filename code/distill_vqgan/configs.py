import torch 

vqmodel_config = {
    "embed_dim": 3,
    "n_embed": 8192,
    "ddconfig" :{
        "double_z": False,
        "z_channels": 3,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult":(1,2,4),
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0
    },
    "lossconfig": torch.nn.Identity
}

distilla_fmri_loss_config = {
    "disc_conditional": False,
    "disc_in_channels": 3,
    "codebook_weight": 1.0,
    "disc_start": 500,
    # "distilla_weight": 1.0
}
