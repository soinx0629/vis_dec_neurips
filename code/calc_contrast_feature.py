
import torch
import pickle
from torchvision.models import resnet50, ResNet50_Weights, ViT_H_14_Weights, vit_h_14
from torchvision.models.feature_extraction import create_feature_extractor
import argparse
import heapq
import os
from tqdm import tqdm
from dataset import create_BOLD5000_dataset_classify, create_Kamitani_dataset_distill

def load_vit(model):
    weights = ViT_H_14_Weights.DEFAULT
    model = vit_h_14(weights=weights)
    preprocess = weights.transforms()
    img_feature_extractor = create_feature_extractor(model, return_nodes={f'encoder.layers.encoder_layer_31.mlp': 'encoder.layers.encoder_layer_31.mlp'}).to('cuda:0').eval()

def get_args_parser():
    parser = argparse.ArgumentParser('MAE finetuning on Test fMRI', add_help=False)

    # Training Parameters
    parser.add_argument('--dataset', type=str, default='BOLD5000')
    parser.add_argument('--fmri_path', type=str, default='./data/BOLD5000')
    parser.add_argument('--subject', type=str, default='CSI3')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--include_nonavg_test', type=bool, default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./contrast_featues')
    parser.add_argument('--calc_nearest', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='vit')
                        
    return parser.parse_args()

def create_feature(thedataset, device, img_feature_extractor, preprocess, num_contrasts=20, calc_nearest=True, model_type='vit'):
    print('wtf', len(thedataset))
    imgname2idx = {}
    all_img_features = []
    all_fmri_features = []
    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    for data_dict in thedataset:
        images = torch.FloatTensor(data_dict['image']).unsqueeze(0)
        images = images.swapaxes(1, 3)
        all_fmri_features.append(data_dict['fmri'])
        # imgname2idx[data_dict['image_name']] = data_id
        if torch.is_nonzero(images.sum(dim=(0,1,2,3))):
        # valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                if model_type == 'resnet':
                    img_features = img_feature_extractor(preprocess(images).to(device))['layer2'].mean(dim=(2,3))
                elif model_type == 'vit':
                    img_features = img_feature_extractor(preprocess(images).to(device))['encoder.layers.encoder_layer_31.mlp'].mean(dim=(1))
                # print(img_features.shape)
                imgname2idx[data_dict['image_name']] = len(all_img_features)
                all_img_features.append(img_features)
        else:
            print('invalid image', data_dict['image_name'])

    all_img_features = torch.stack(all_img_features).squeeze(1)
    all_fmri_features = torch.stack(all_fmri_features).squeeze(1)
    print('all_img_features shape', all_img_features.shape, 'all_fmri_features shape', all_fmri_features.shape)
    
    if calc_nearest:
        image_rsa = []
        all_positive_sample_index = []
        all_negative_sample_index = []
        for img_features_i in tqdm(all_img_features):
            image_rsa_i = []
            for img_features_j in all_img_features:
                sim = cos_sim(img_features_i, img_features_j)
                image_rsa_i.append(sim)
            
            image_rsa_i = torch.FloatTensor(image_rsa_i)
            image_rsa.append(image_rsa_i)

            # all_positive_sample_index.append(heapq.nlargest(num_contrasts, range(len(image_rsa_i)), image_rsa_i.cpu().numpy().take))
            # all_negative_sample_index.append(heapq.nsmallest(num_contrasts, range(len(image_rsa_i)), image_rsa_i.cpu().numpy().take))
            
        img_rsa = torch.stack(image_rsa)
        print('img_rsa shape', img_rsa.shape)
        all_positive_sample_index = torch.topk(img_rsa, num_contrasts, largest=True, sorted=True).indices
        all_negative_sample_index = torch.topk(img_rsa, num_contrasts, largest=False, sorted=True).indices
        # print('img_rsa shape', img_rsa.shape)

        all_positive_sample_index = torch.Tensor(all_positive_sample_index)
        all_negative_sample_index = torch.Tensor(all_negative_sample_index)
        # all_positive_sample_index = torch.topk(all_positive_sample_index, k=num_contrasts, dim=1, largest=True).indices
        # all_negative_sample_index = torch.topk(all_negative_sample_index, k=num_contrasts, dim=1, largest=False).indices
        print('all_positive_sample_index shape', all_positive_sample_index.shape, 
            'all_negative_sample_index shape', all_negative_sample_index.shape)

        return all_img_features, all_fmri_features, imgname2idx, img_rsa, all_positive_sample_index, all_negative_sample_index
    else:
        return all_img_features, all_fmri_features, imgname2idx

def main(args):

    device = torch.device(f'cuda:{args.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

    print('Loading BOLD5000 dataset...')

    if args.dataset == 'BOLD5000':
        train_set, test_set = create_BOLD5000_dataset_classify(path=args.fmri_path, patch_size=args.patch_size, 
                fmri_transform=torch.FloatTensor, subjects=[args.subject], include_nonavg_test=args.include_nonavg_test)
    else:
        train_set, test_set = create_Kamitani_dataset_distill(path=args.fmri_path, patch_size=args.patch_size,
                                subjects=[args.subject], fmri_transform=torch.FloatTensor, include_nonavg_test=args.include_nonavg_test,
                                return_image_name=True)
        
    print('Train set size: ', len(train_set), 'Test set size: ', len(test_set))
    
    if args.model_type == 'vit':
        print('Loading vit as feature extractor...')
        weights = ViT_H_14_Weights.DEFAULT
        model = vit_h_14(weights=weights)
        preprocess = weights.transforms()
        img_feature_extractor = create_feature_extractor(model, return_nodes={f'encoder.layers.encoder_layer_31.mlp': 'encoder.layers.encoder_layer_31.mlp'}).to(device).eval()
    elif args.model_type == 'resnet':
        print('Loading ResNet50 as feature extractor...')
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        m = resnet50(weights=weights)   
        img_feature_extractor = create_feature_extractor(m, return_nodes={f'layer2': 'layer2'}).to(device).eval()
        for param in img_feature_extractor.parameters():
            param.requires_grad = False

        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)   

    if args.calc_nearest:
        print('Creating train set features and rsa...')

        tr_img_features, tr_fmri_features,tr_imgname2idx, tr_img_rsa, tr_positive_sample_index, tr_negative_sample_index = create_feature(train_set, device,
                                                                                                            img_feature_extractor, preprocess)

        print('Creating test set features and rsa...')  
        te_img_features, te_fmri_features, te_imgname2idx, te_img_rsa, te_positive_sample_index, te_negative_sample_index = create_feature(test_set, device, 
                                                                                                            img_feature_extractor, preprocess)

        print('Saving features...')                                                                                        

        torch.save(tr_img_features.detach().cpu(), args.output_dir + f'/tr_img_features_{args.subject}.pt')
        torch.save(tr_fmri_features.cpu(), args.output_dir + f'/tr_fmri_features_{args.subject}.pt')
        torch.save(tr_img_rsa.cpu(), args.output_dir + f'/tr_img_rsa_{args.subject}.pt')
        torch.save(tr_positive_sample_index.cpu(), args.output_dir + f'/tr_positive_sample_index_{args.subject}.pt')
        torch.save(tr_negative_sample_index.cpu(), args.output_dir + f'/tr_negative_sample_index_{args.subject}.pt')
        pickle.dump(tr_imgname2idx, open(args.output_dir + f'/tr_imgname2idx_{args.subject}.pkl', 'wb'))

        torch.save(te_img_features.detach().cpu(), args.output_dir + f'/te_img_features_{args.subject}.pt')
        torch.save(te_fmri_features.cpu(), args.output_dir + f'/te_fmri_features_{args.subject}.pt')
        torch.save(te_img_rsa.cpu(), args.output_dir + f'/te_img_rsa_{args.subject}.pt')
        torch.save(te_positive_sample_index.cpu(), args.output_dir + f'/te_positive_sample_index_{args.subject}.pt')
        torch.save(te_negative_sample_index.cpu(), args.output_dir + f'/te_negative_sample_index_{args.subject}.pt')
        pickle.dump(te_imgname2idx, open(args.output_dir + f'/te_imgname2idx_{args.subject}.pkl', 'wb'))

    else:
        print('Creating train set features...')
        tr_img_features, tr_fmri_features,tr_imgname2idx = create_feature(train_set, device, img_feature_extractor, preprocess, calc_nearest=args.calc_nearest, model_type=args.model_type)
        print('Creating test set features...')
        te_img_features, te_fmri_features, te_imgname2idx = create_feature(test_set, device, img_feature_extractor, preprocess,calc_nearest=args.calc_nearest, model_type=args.model_type)
        print('Saving features in shape ', tr_img_features.shape, tr_fmri_features.shape, te_img_features.shape, te_fmri_features.shape)

        torch.save(tr_img_features.detach().cpu(), args.output_dir + f'/tr_img_features_{args.model_type}_{args.dataset}_{args.subject}.pt')
        torch.save(tr_fmri_features.cpu(), args.output_dir + f'/tr_fmri_features_{args.model_type}_{args.dataset}_{args.subject}.pt')
        pickle.dump(tr_imgname2idx, open(args.output_dir + f'/tr_imgname2idx_{args.model_type}_{args.dataset}_{args.subject}.pkl', 'wb'))
        
        torch.save(te_img_features.detach().cpu(), args.output_dir + f'/te_img_features_{args.model_type}_{args.dataset}_{args.subject}.pt')
        torch.save(te_fmri_features.cpu(), args.output_dir + f'/te_fmri_features_{args.model_type}_{args.dataset}_{args.subject}.pt')
        pickle.dump(te_imgname2idx, open(args.output_dir + f'/te_imgname2idx_{args.model_type}_{args.dataset}_{args.subject}.pkl', 'wb'))
        
        
if __name__ == '__main__':
    args = get_args_parser()
    main(args)
    