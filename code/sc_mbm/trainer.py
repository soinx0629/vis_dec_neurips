import math, sys
import torch
import sc_mbm.utils as ut
from torch._six import inf
import numpy as np
import time

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                if type(parameters) == list:
                    norm0 = torch.nn.utils.clip_grad_norm_(parameters[0], clip_grad)
                    norm1 = torch.nn.utils.clip_grad_norm_(parameters[1], clip_grad)
                    norm = norm0 + norm1
                else:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                # print(norm)
            else:
                self._scaler.unscale_(optimizer)
                if type(parameters) == list:
                    norm0 = get_grad_norm_(parameters[0])
                    norm1 = get_grad_norm_(parameters[1])
                    norm = norm0 + norm1
                else:
                    norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    # print(total_norm)
    return total_norm


def train_one_epoch_cross(model, model_image, data_loader, optimizer, device, epoch, 
                        loss_scaler,log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                        img_feature_extractor=None, preprocess=None, optimizer_img=None, loss_scaler_img=None,
                        fmri_recon_weight=1.0, img_recon_weight=1.0,):
    model.train(True)
    model_image.train(True)
    optimizer.zero_grad()
    total_loss = []
    total_loss_image = []
    total_cor = []
    total_cor_image = []
    # total_loss_fmri = []
    accum_iter = config.accum_iter
    for data_iter_step, (data_dcit) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples = data_dcit['fmri']
        
        images = data_dcit['image']
        valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)

        img_prep = img_feature_extractor(images=images, return_tensors="pt")
        img_prep["pixel_values"] = img_prep["pixel_values"].to(device)
        # print(img_prep["pixel_values"].device)
        # print('model', model_image.device)
        # print(img_prep["pixel_values"].shape)

        samples = samples.to(device)
        # img_features = img_features.to(device)
        # import ipdb
        # ipdb.set_trace()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            # reconstruct fmri
            img_support = model_image(pixel_values=img_prep["pixel_values"], given_mask_ratio=0, encoder_only=True)
            # print('img supprto', img_support.last_hidden_state.shape)
            loss_fmri_recon, pred, _ = model(samples, valid_idx=valid_idx, mask_ratio=config.mask_ratio, image_support=img_support.last_hidden_state)
            # reconstruct image
            fmri_support = model(samples, mask_ratio=0, encoder_only=True)
            # print('fmri_support ', fmri_support.shape)
            img_recons_output = model_image(pixel_values=img_prep["pixel_values"], given_mask_ratio=config.img_mask_ratio, fmri_support=fmri_support)

        # loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        # optimizer.step()

        loss = fmri_recon_weight*loss_fmri_recon + img_recon_weight*img_recons_output.loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        if optimizer_img is None or loss_scaler_img is None:
            loss_scaler(loss, optimizer, parameters=[model.parameters(), model_image.parameters()], clip_grad=config.clip_grad)
        else:
            loss_scaler(loss_fmri_recon, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)
            loss_scaler_img(img_recons_output.loss, optimizer_img, parameters=model_image.parameters(), clip_grad=config.clip_grad)
        # loss_scaler(img_recons_output.loss, optimizer, parameters=model_image.parameters(), clip_grad=config.clip_grad)

        # if (data_iter_step + 1) % accum_iter == 0:
        # cal the cor
        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        pred = model_without_ddp.unpatchify(pred)
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p, s],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        cor_image = img_recons_output.corr
        optimizer.zero_grad()

        total_loss.append(loss_fmri_recon.item())
        total_loss_image.append(img_recons_output.loss.item())
        total_cor.append(cor)
        total_cor_image.append(cor_image)

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step_fmri', np.mean(total_loss), step=epoch)
        log_writer.log('train_loss_step_image', np.mean(total_loss_image), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        log_writer.log('cor_fmri', np.mean(total_cor), step=epoch)
        log_writer.log('cor_image', np.mean(cor_image), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] train loss fmri: {np.mean(total_loss)} train loss image: {np.mean(total_loss_image)}')
        print(f'[Epoch {epoch}] train corr fmri: {np.mean(total_cor)} train corr image: {np.mean(total_cor_image)}')

    return np.mean(total_cor_image)

def nxn_cos_sim(A, B, dim=1, eps=1e-8):
      numerator = A @ B.T
      A_l2 = torch.mul(A, A).sum(axis=dim)
      B_l2 = torch.mul(B, B).sum(axis=dim)
      denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
      return torch.div(numerator, denominator)

def eval_wordemb_hit(pred_emb, true_label, label2emb_dict):
    all_labels = []
    all_embs = []
    for kk, vv in label2emb_dict.items():
        all_labels.append(kk)
        all_embs.append(vv)

    all_embs = torch.FloatTensor(np.array(all_embs))
    # sim = pred_emb.float() @ all_embs.transpose(-1,-2)
    sim = nxn_cos_sim(pred_emb.float(), all_embs)
    # print('sim shape is ', sim.shape, pred_emb.shape, all_embs.shape)
    top_sim_ind = torch.topk(sim, k=5, dim=-1, largest=True, sorted=True).indices
    pred_labels = [[all_labels[jj] for jj in ii] for ii in top_sim_ind]
    hit = 0.0
    for ii, jj in zip(pred_labels, true_label):
        # print(jj, ii)
        if jj in ii:
            hit += 1.0

    return hit/len(true_label)
    

def train_one_epoch_contrast(model, data_loader, optimizer, device, epoch, loss_scaler,
                            log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                            img_feature_extractor=None, preprocess=None,
                            train_contrast_feature_dict=None, test_contrast_feature_dict=None, 
                            num_sel_pos_contrast=1, num_sel_neg_contrast=4, num_voxels=4192, do_test=False, 
                            all_distill_target=None, imgname2idx=None, return_fmri_map=False):
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    total_contrast_loss = []
    total_mask_loss = []
    total_distill_loss = []
   
    step_loss = []
    step_contrast_loss = []
    step_mask_loss = []
    step_distill_loss = []

    accum_iter = config.accum_iter

    total_cor = []
    step_cor = []


    # if 'tr_positive_sample_index' in contrast_feature_dict:
    if config.do_sup_contrast:
        all_positive_sample_index = train_contrast_feature_dict['tr_positive_sample_index']
        all_negative_sample_index = train_contrast_feature_dict['tr_negative_sample_index']
        # imgname2idx = train_contrast_feature_dict['tr_imgname2idx']
        all_fmri_features = train_contrast_feature_dict['tr_fmri_features']

        if do_test:
            all_positive_sample_index = test_contrast_feature_dict['te_positive_sample_index']
            all_negative_sample_index = test_contrast_feature_dict['te_negative_sample_index']
            # test_imgname2idx = test_contrast_feature_dict['te_imgname2idx']
            all_fmri_features = test_contrast_feature_dict['te_fmri_features']
    else:
        all_positive_sample_index = None
        all_negative_sample_index = None
        # imgname2idx = None
        all_fmri_features = None

    # if return_fmri_map:
    all_fmri_map = []
    all_image_labels = []

    for data_iter_step, (data_dcit) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples = data_dcit['fmri']
        
        img_features = None
        valid_idx = None

        if img_feature_extractor is not None:
            images = data_dcit['image']
            
            valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                img_features = img_feature_extractor(preprocess(images[valid_idx]).to(device))['layer2']

        samples = samples.to(device)
        # img_features = img_features.to(device)

        if config.do_sup_contrast or config.do_distill_loss or config.do_distill_contrast:
            if config.distill_feat == 'vit' or config.distill_feat == 'ldm':
                image_names = data_dcit['image_name']
                image_ids = [imgname2idx[img_name.item()] for img_name in image_names]
            elif config.distill_feat == 'wordemb':
                image_labels = data_dcit['image_class_name']
                all_image_labels.append(image_labels)

        if config.do_sup_contrast:
            positive_sample_indexes = all_positive_sample_index[image_ids]
            negative_sample_indexes = all_negative_sample_index[image_ids]
            # print(type(positive_sample_indexes), positive_sample_indexes, len(positive_sample_indexes))

            positive_fmri_feats = []
            negative_fmri_feats = []

            positive_fmri_feats = [all_fmri_features[sindex[1:num_sel_pos_contrast+1]] for sindex in positive_sample_indexes]
            negative_fmri_feats = [all_fmri_features[sindex[:num_sel_neg_contrast]] for sindex in negative_sample_indexes]

            positive_fmri_feats = torch.stack(positive_fmri_feats)
            negative_fmri_feats = torch.stack(negative_fmri_feats)
            
            positive_fmri_feats= torch.FloatTensor(np.pad(positive_fmri_feats.squeeze(1), ((0,0), (0, num_voxels - positive_fmri_feats.shape[-1])), 'wrap'))
            negative_fmri_feats= torch.FloatTensor(np.pad(negative_fmri_feats, ((0,0), (0,0), (0, num_voxels - negative_fmri_feats.shape[-1])), 'wrap'))

            positive_fmri_feats.to(device)
            negative_fmri_feats.to(device)

        else:
            positive_fmri_feats = None
            negative_fmri_feats = None

        if config.do_distill_loss or config.do_distill_contrast:
            # print('yes do distill contrast')
            if config.distill_feat == 'vit' or config.distill_feat == 'ldm':
                distill_pos_sample = all_distill_target[image_ids].to(device)
            else:
                distill_pos_sample = torch.FloatTensor(np.array([all_distill_target[lab] for lab in image_labels])).to(device)
        else:
            distill_pos_sample = None


        # print(samples.shape, negative_fmri_feats.shape, positive_fmri_feats.shape)

        # positive_fmri_feats = all_fmri_features[positive_sample_indexes]
        # negative_fmri_feats = all_fmri_features[negative_sample_indexes]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _, mask_loss, contrast_loss, distill_loss = model(samples, mask_ratio=config.mask_ratio, 
                                  pos_sample=positive_fmri_feats, neg_sample=negative_fmri_feats, 
                                  distill_pos_sample=distill_pos_sample, dropout=True)
            # all_fmri_map.append(fmri_map.to('cpu'.detach()))
        # loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        # optimizer.step()



        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        # if (data_iter_step + 1) % accum_iter == 0:
        # cal the cor
        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        pred = model_without_ddp.unpatchify(pred)
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p, s],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        optimizer.zero_grad()

        total_loss.append(loss_value)
        total_cor.append(cor)
        total_contrast_loss.append(contrast_loss.item())
        total_mask_loss.append(mask_loss.item())
        total_distill_loss.append(distill_loss.item())

        step_loss.append(loss_value)
        step_cor.append(cor)
        step_contrast_loss.append(contrast_loss.item())
        step_mask_loss.append(mask_loss.item())
        step_distill_loss.append(distill_loss.item())

        logger_flag = 'test' if do_test else 'train'
        logger_step = 2 if do_test else 10

        if data_iter_step % logger_step == 0:
            if config.local_rank == 0:
                print('epoch: {}, data_iter_step: {}, total loss step: {}, contrast loss step: {}, distill loss step {}, mask loss step {}, cor step: {}'.format(
                    epoch, data_iter_step, loss_value, contrast_loss.item(), distill_loss.item(), mask_loss.item(), cor))
            if log_writer is not None:
                lr = optimizer.param_groups[0]["lr"]
                step_count = data_iter_step + epoch * len(data_loader)
                log_writer.log(f'{logger_flag}_loss_step', np.mean(step_loss), step=step_count)
                log_writer.log(f'{logger_flag}_contrast_loss_step', np.mean(step_contrast_loss), step=step_count)
                log_writer.log(f'{logger_flag}_distill_loss_step', np.mean(step_distill_loss), step=step_count)
                log_writer.log(f'{logger_flag}_mask_loss_step', np.mean(step_mask_loss), step=step_count)
                log_writer.log('lr_step', lr, step=step_count)
                log_writer.log('cor_step', np.mean(step_cor), step=step_count)
                if start_time is not None:
                    log_writer.log('time step(min)', (time.time() - start_time)/60.0, step=step_count)

            if data_iter_step > 0:
                step_loss = []
                step_cor = []
                step_contrast_loss = []
                step_mask_loss = []
                step_distill_loss = []

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log(f'{logger_flag}_loss_epoch', np.mean(total_loss), step=epoch)
        log_writer.log(f'{logger_flag}_contrast_loss_epoch', np.mean(total_contrast_loss), step=epoch)
        log_writer.log(f'{logger_flag}_distill_loss_epoch', np.mean(total_distill_loss), step=epoch)
        log_writer.log(f'{logger_flag}_mask_loss_epoch', np.mean(total_mask_loss), step=epoch)
        log_writer.log('lr_epoch', lr, step=epoch)
        log_writer.log('cor_epoch', np.mean(total_cor), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
        
    if config.local_rank == 0:        
        print(f'[Epoch {epoch} {logger_flag}] loss: {np.mean(total_loss)} contrast loss: {np.mean(total_contrast_loss)} cor: {np.mean(total_cor)} output_path_ifsaved: {config.output_path}')

    return np.mean(total_cor)


def eval_one_epoch_contrast(model, data_loader, device, epoch,
                            log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                            img_feature_extractor=None, preprocess=None,
                            train_contrast_feature_dict=None, test_contrast_feature_dict=None, 
                            num_sel_pos_contrast=1, num_sel_neg_contrast=4, num_voxels=4192, do_test=False,
                            all_distill_target=None, imgname2idx=None, return_fmri_map=False):
    model.eval()

    total_loss = []
    total_contrast_loss = []
    total_mask_loss = []
    total_distill_loss = []
   
    step_loss = []
    step_contrast_loss = []
    step_mask_loss = []
    step_distill_loss = []

    total_cor = []
    step_cor = []

    accum_iter = config.accum_iter

    if config.do_sup_contrast:

        all_positive_sample_index = test_contrast_feature_dict['te_positive_sample_index']
        all_negative_sample_index = test_contrast_feature_dict['te_negative_sample_index']
        # imgname2idx = test_contrast_feature_dict['te_imgname2idx']
        all_fmri_features = test_contrast_feature_dict['te_fmri_features']

    pred_fmri_map = []

    for data_iter_step, (data_dcit) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        
        samples = data_dcit['fmri']
        
        img_features = None
        valid_idx = None

        if img_feature_extractor is not None:
            images = data_dcit['image']
            
            valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                img_features = img_feature_extractor(preprocess(images[valid_idx]).to(device))['layer2']

        samples = samples.to(device)
        # img_features = img_features.to(device)

        if config.do_sup_contrast or config.do_distill_loss or config.do_distill_contrast:
            if config.distill_feat == 'vit' or config.distill_feat == 'ldm':
                image_names = data_dcit['image_name']
                image_ids = [imgname2idx[img_name.item()] for img_name in image_names]
            elif config.distill_feat == 'wordemb':
                image_labels = data_dcit['image_class_name']
                # all_image_labels.append(image_labels)


        if config.do_sup_contrast:
            positive_sample_indexes = all_positive_sample_index[image_ids]
            negative_sample_indexes = all_negative_sample_index[image_ids]
            # print(type(positive_sample_indexes), positive_sample_indexes, len(positive_sample_indexes))

            positive_fmri_feats = []
            negative_fmri_feats = []

            positive_fmri_feats = [all_fmri_features[sindex[1:num_sel_pos_contrast+1]] for sindex in positive_sample_indexes]
            negative_fmri_feats = [all_fmri_features[sindex[:num_sel_neg_contrast]] for sindex in negative_sample_indexes]

            positive_fmri_feats = torch.stack(positive_fmri_feats)
            negative_fmri_feats = torch.stack(negative_fmri_feats)
            
            positive_fmri_feats = torch.FloatTensor(np.pad(positive_fmri_feats.squeeze(1), ((0,0), (0, num_voxels - positive_fmri_feats.shape[-1])), 'wrap'))
            negative_fmri_feats = torch.FloatTensor(np.pad(negative_fmri_feats, ((0,0), (0,0), (0, num_voxels - negative_fmri_feats.shape[-1])), 'wrap'))

            # print(samples.shape, negative_fmri_feats.shape, positive_fmri_feats.shape)

            # positive_fmri_feats = all_fmri_features[positive_sample_indexes]
            # negative_fmri_feats = all_fmri_features[negative_sample_indexes]
            positive_fmri_feats.to(device)
            negative_fmri_feats.to(device)
        else:
            positive_fmri_feats = None
            negative_fmri_feats = None

        if config.do_distill_loss or config.do_distill_contrast:
            if config.distill_feat == 'vit' or config.distill_feat == 'ldm':
                distill_pos_sample = all_distill_target[image_ids].to(device)
            else:
                distill_pos_sample = torch.FloatTensor(np.array([all_distill_target[lab] for lab in image_labels])).to(device)
        else:
            distill_pos_sample = None

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                if config.wordembed and return_fmri_map:
                    loss, pred, _, mask_loss, contrast_loss, distill_loss, fmri_map = model(samples, mask_ratio=config.mask_ratio, 
                                        pos_sample=positive_fmri_feats, neg_sample=negative_fmri_feats,
                                        distill_pos_sample=distill_pos_sample, dropout=False, return_fmri_map=return_fmri_map)
                    hit_score = eval_wordemb_hit(fmri_map.to('cpu').detach(), image_labels, all_distill_target)
                    print(f'epoch {epoch} iteration {data_iter_step} hit score: {hit_score}')
                else:
                    loss, pred, _, mask_loss, contrast_loss, distill_loss = model(samples, mask_ratio=config.mask_ratio, 
                                        pos_sample=positive_fmri_feats, neg_sample=negative_fmri_feats,
                                        distill_pos_sample=distill_pos_sample, dropout=False)
            

                
        # loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        # optimizer.step()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        # loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        # if (data_iter_step + 1) % accum_iter == 0:
        # cal the cor
        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        pred = model_without_ddp.unpatchify(pred)
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p, s],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        # optimizer.zero_grad()

        total_loss.append(loss_value)
        total_cor.append(cor)
        total_contrast_loss.append(contrast_loss.item())
        total_mask_loss.append(mask_loss.item())
        total_distill_loss.append(distill_loss.item())

        step_loss.append(loss_value)
        step_cor.append(cor)
        step_contrast_loss.append(contrast_loss.item())
        step_mask_loss.append(mask_loss.item())
        step_distill_loss.append(distill_loss.item())

        if data_iter_step % 10 == 0:
            print('epoch: {}, data_iter_step: {}, total loss step: {}, contrast loss step: {}, cor step: {}'.format(epoch, data_iter_step, loss_value, 
                                                                                          contrast_loss.item(), cor))
            if log_writer is not None:
                # lr = optimizer.param_groups[0]["lr"]
                step_count = data_iter_step + epoch * len(data_loader)
                log_writer.log('test_loss_step', np.mean(step_loss), step=step_count)
                log_writer.log('test_contrast_loss_step', np.mean(step_contrast_loss), step=step_count)
                log_writer.log('test_cor_step', np.mean(step_cor), step=step_count)
                log_writer.log('test_contrast_loss_step', np.mean(step_contrast_loss), step=step_count)
                log_writer.log('test_distill_loss_step', np.mean(step_distill_loss), step=step_count)
                log_writer.log('test_mask_loss_step', np.mean(step_mask_loss), step=step_count)
                if start_time is not None:
                    log_writer.log('time step(min)', (time.time() - start_time)/60.0, step=step_count)

            if data_iter_step > 0:
                step_loss = []
                step_cor = []
                step_contrast_loss = []
                step_distill_loss = []
                step_mask_loss = []

    if log_writer is not None and config.local_rank == 0:
        # lr = optimizer.param_groups[0]["lr"]
        log_writer.log('test_loss_epoch', np.mean(total_loss), step=epoch)
        log_writer.log('test_contrast_loss_epoch', np.mean(total_contrast_loss), step=epoch)
        log_writer.log('test_epoch', np.mean(total_cor), step=epoch)
        log_writer.log('test_contrast_loss_step', np.mean(step_contrast_loss), step=step_count)
        log_writer.log('test_distill_loss_step', np.mean(step_distill_loss), step=step_count)
        log_writer.log('test_mask_loss_step', np.mean(step_mask_loss), step=step_count)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
        
    # if config.local_rank == 0:        
    print(f'[Eval Epoch {epoch}] loss: {np.mean(total_loss)} contrast loss: {np.mean(total_contrast_loss)} cor: {np.mean(total_cor)}',
          f'distill loss: {np.mean(total_distill_loss)} mask loss: {np.mean(total_mask_loss)}')
    
    return np.mean(total_loss), np.mean(total_contrast_loss), np.mean(total_cor)




def eval_one_epoch_cross(model, model_image, data_loader, device, epoch, log_writer=None, config=None, 
                        start_time=None, model_without_ddp=None, img_feature_extractor=None):
    model.eval()
    model_image.eval()
    total_loss = []
    total_loss_image = []
    total_cor = []
    total_cor_image = []
    # total_loss_fmri = []
    accum_iter = config.accum_iter
    for data_iter_step, (data_dcit) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        samples = data_dcit['fmri']
        
        images = data_dcit['image']
        valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)

        img_prep = img_feature_extractor(images=images, return_tensors="pt")
        # print(img_prep["pixel_values"].shape)
        img_prep["pixel_values"] = img_prep["pixel_values"].to(device)

        samples = samples.to(device)
        # img_features = img_features.to(device)

        with torch.no_grad():
            # reconstruct fmri
            img_support = model_image(pixel_values=img_prep["pixel_values"], given_mask_ratio=0, encoder_only=True)
            # print('img supprto', img_support.last_hidden_state.shape)
            loss_fmri_recon, pred, _ = model(samples, valid_idx=valid_idx, mask_ratio=config.mask_ratio, image_support=img_support.last_hidden_state)
            # reconstruct image
            fmri_support = model(samples, mask_ratio=0, encoder_only=True)
            # print('fmri_support ', fmri_support.shape)
            img_recons_output = model_image(pixel_values=img_prep["pixel_values"], given_mask_ratio=config.img_mask_ratio, fmri_support=fmri_support)


        loss = loss_fmri_recon + img_recons_output.loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss_scaler(img_recons_output.loss, optimizer, parameters=model_image.parameters(), clip_grad=config.clip_grad)

        # if (data_iter_step + 1) % accum_iter == 0:
        # cal the cor
        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        pred = model_without_ddp.unpatchify(pred)
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p, s],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        cor_image = img_recons_output.corr

        total_loss.append(loss_fmri_recon.item())
        total_loss_image.append(img_recons_output.loss.item())
        total_cor.append(cor)
        total_cor_image.append(cor_image)

    if log_writer is not None:
        log_writer.log('test_loss_step_fmri', np.mean(total_loss), step=epoch)
        log_writer.log('test_loss_step_image', np.mean(total_loss_image), step=epoch)
        log_writer.log('test_cor_fmri', np.mean(total_cor), step=epoch)
        log_writer.log('test_cor_image', np.mean(cor_image), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] test loss fmri: {np.mean(total_loss)} test loss image: {np.mean(total_loss_image)}')
        print(f'[Epoch {epoch}] test corr fmri: {np.mean(total_cor)} test corr image: {np.mean(total_cor_image)}')


    return np.mean(total_cor_image)