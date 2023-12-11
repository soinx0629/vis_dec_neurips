from mae_for_image import ViTMAEForPreTraining
from transformers import AutoFeatureExtractor

mae_image_model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
mae_image_model.vit.eval()
for param in mae_image_model.vit.parameters():
    param.requires_grad = False

mae_image_model.decoder.train()
for param in mae_image_model.decoder.parameters():
    param.requires_grad = True

# reconstruct image with the support of fmri data
feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/vit-mae-large')
inputs = feature_extractor(images=image, return_tensors="pt")
image_reconstruct_output = mae_image_model(pixel_values=inputs["pixel_values"], given_mask_ratio=0.85, fmri_support=fmri_support)

# reconstruct fmri data with the support of image 
image_support = mae_image_model(pixel_values=inputs["pixel_values"], given_mask_ratio=0)



