import torch
import torch.nn as nn
import numpy as np
from .cc_function import get_rank, get_world_size, AverageMeter, \
    allreduce_tensor, save_results_more, reduce_tensor

def build_validator(config):
    name = config.test.get("validator", None)
    if not name:
        name = "ValidatorBase"
    return eval(name)()

class ValidatorBase:

    def validate(self, config, testloader, model, writer_dict, device,
                 num_patches, img_vis_dir, mean, std):

        rank = get_rank()
        world_size = get_world_size()
        model.eval()
        avg_loss = AverageMeter()
        cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(),
                      'nae': AverageMeter(), 'acc1': AverageMeter()}
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                # if _>100:
                #     break
                image, label, _, name = batch
                image = image.to(device)
                for i in range(len(label)):
                    label[i] = label[i].to(device)
                # result = model(image, label, 'val')
                result = self.patch_forward(config, model, image, label, num_patches, 'val')

                losses = result['losses']
                pre_den = result['pre_den']['1']
                gt_den = result['gt_den']['1']

                #    -----------Counting performance------------------
                gt_count, pred_cnt = label[0].sum(), pre_den.sum()

                # print(f" rank: {rank} gt: {gt_count} pre: {pred_cnt}")
                s_mae = torch.abs(gt_count - pred_cnt)

                s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))

                allreduce_tensor(s_mae)
                allreduce_tensor(s_mse)
                # acc1 = reduce_tensor(result['acc1']['error']/(result['acc1']['gt']+1e-10))
                reduced_loss = reduce_tensor(losses)
                # print(f" rank: {rank} mae: {s_mae} mse: {s_mse}"
                #       f"loss: {reduced_loss}")
                avg_loss.update(reduced_loss.item())
                # cnt_errors['acc1'].update(acc1)
                cnt_errors['mae'].update(s_mae.item())
                cnt_errors['mse'].update(s_mse.item())

                s_nae = (torch.abs(gt_count - pred_cnt) / (gt_count + 1e-10))
                allreduce_tensor(s_nae)
                cnt_errors['nae'].update(s_nae.item())

                if rank == 0:
                    if idx % 20 == 0:
                        # acc1 = cnt_errors['acc1'].avg/world_size
                        # print( f'acc1:{acc1}')
                        image = image[0]
                        for t, m, s in zip(image, mean, std):
                            t.mul_(s).add_(m)
                        save_results_more(name[0], img_vis_dir, image.cpu().data, \
                                          pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
                                          pred_cnt.item(), gt_count.item())
        print_loss = avg_loss.average() / world_size

        mae = cnt_errors['mae'].avg / world_size
        mse = np.sqrt(cnt_errors['mse'].avg / world_size)
        nae = cnt_errors['nae'].avg / world_size

        if rank == 0:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', print_loss, global_steps)
            writer.add_scalar('valid_mae', mae, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        return print_loss, mae, mse, nae

    def patch_forward(self, config, model, img, dot_map, num_patches, mode):
        # crop the img and gt_map with a max stride on x and y axis
        # size: HW: __C_NWPU.TRAIN_SIZE
        # stack them with a the batchsize: __C_NWPU.TRAIN_BATCH_SIZE
        crop_imgs = []
        crop_dots, crop_masks = {}, {}

        crop_dots['1'], crop_dots['2'], crop_dots['4'], crop_dots['8'] = [], [], [], []
        crop_masks['1'], crop_masks['2'], crop_masks['4'], crop_masks['8'] = [], [], [], []
        b, c, h, w = img.shape

        crop_size = config.test.get('crop_size', (768, 1024))  # default (768, 1024)
        rh, rw = crop_size

        # support for multi-scale patch forward
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                for res_i in range(len(dot_map)):
                    gis_, gie_ = gis // 2 ** res_i, gie // 2 ** res_i
                    gjs_, gje_ = gjs // 2 ** res_i, gje // 2 ** res_i
                    crop_dots[str(2 ** res_i)].append(dot_map[res_i][:, gis_:gie_, gjs_:gje_])
                    mask = torch.zeros_like(dot_map[res_i]).cpu()
                    mask[:, gis_:gie_, gjs_:gje_].fill_(1.0)
                    crop_masks[str(2 ** res_i)].append(mask)

        crop_imgs = torch.cat(crop_imgs, dim=0)
        for k, v in crop_dots.items():
            crop_dots[k] = torch.cat(v, dim=0)
        for k, v in crop_masks.items():
            crop_masks[k] = torch.cat(v, dim=0)

        # forward may need repeatng
        crop_losses = []
        crop_preds = {}
        crop_labels = {}
        crop_labels['1'], crop_labels['2'], crop_labels['4'], crop_labels['8'] = [], [], [], []
        crop_preds['1'], crop_preds['2'], crop_preds['4'], crop_preds['8'] = [], [], [], []
        nz, bz = crop_imgs.size(0), num_patches
        keys_pre = None

        for i in range(0, nz, bz):
            gs, gt = i, min(nz, i + bz)
            result = model(crop_imgs[gs:gt], [crop_dots[k][gs:gt] for k in crop_dots.keys()],
                           mode)
            crop_pred = result['pre_den']
            crop_label = result['gt_den']

            keys_pre = result['pre_den'].keys()
            for k in keys_pre:
                crop_preds[k].append(crop_pred[k].cpu())
                crop_labels[k].append(crop_label[k].cpu())

            crop_losses.append(result['losses'].mean())

        for k in keys_pre:
            crop_preds[k] = torch.cat(crop_preds[k], dim=0)
            crop_labels[k] = torch.cat(crop_labels[k], dim=0)

        # splice them to the original size

        result = {'pre_den': {}, 'gt_den': {}}

        for res_i, k in enumerate(keys_pre):

            pred_map = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
            labels = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
            idx = 0
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                    gis_, gie_ = gis // 2 ** res_i, gie // 2 ** res_i
                    gjs_, gje_ = gjs // 2 ** res_i, gje // 2 ** res_i

                    pred_map[:, :, gis_:gie_, gjs_:gje_] += crop_preds[k][idx]
                    labels[:, :, gis_:gie_, gjs_:gje_] += crop_labels[k][idx]
                    idx += 1
            # import pdb
            # pdb.set_trace()
            # for the overlapping area, compute average value
            mask = crop_masks[k].sum(dim=0).unsqueeze(0).unsqueeze(0)
            pred_map = (pred_map / mask)
            labels = (labels / mask)
            result['pre_den'].update({k: pred_map})
            result['gt_den'].update({k: labels})
            result.update({'losses': crop_losses[0]})
        return result


class Validator_Adapt_kernel:

    def validate(self, config, testloader, model, writer_dict, device,
                 num_patches, img_vis_dir, mean, std):

        rank = get_rank()
        world_size = get_world_size()
        model.eval()
        avg_loss = AverageMeter()
        cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(),
                      'nae': AverageMeter(), 'acc1': AverageMeter()}
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                # if _>100:
                #     break
                image, label, _, name, *args = batch
                image = image.to(device)
                for i in range(len(label)):
                    label[i] = label[i].to(device)
                # result = model(image, label, 'val')
                result = self.patch_forward(config, model, image, label, num_patches, args, 'val')

                losses = result['losses']
                pre_den = result['pre_den']['1']
                gt_den = result['gt_den']['1']

                #    -----------Counting performance------------------
                gt_count, pred_cnt = label[0].sum(), pre_den.sum()

                # print(f" rank: {rank} gt: {gt_count} pre: {pred_cnt}")
                s_mae = torch.abs(gt_count - pred_cnt)

                s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))

                allreduce_tensor(s_mae)
                allreduce_tensor(s_mse)
                # acc1 = reduce_tensor(result['acc1']['error']/(result['acc1']['gt']+1e-10))
                reduced_loss = reduce_tensor(losses)
                # print(f" rank: {rank} mae: {s_mae} mse: {s_mse}"
                #       f"loss: {reduced_loss}")
                avg_loss.update(reduced_loss.item())
                # cnt_errors['acc1'].update(acc1)
                cnt_errors['mae'].update(s_mae.item())
                cnt_errors['mse'].update(s_mse.item())

                s_nae = (torch.abs(gt_count - pred_cnt) / (gt_count + 1e-10))
                allreduce_tensor(s_nae)
                cnt_errors['nae'].update(s_nae.item())

                if rank == 0:
                    if idx % 20 == 0:
                        # acc1 = cnt_errors['acc1'].avg/world_size
                        # print( f'acc1:{acc1}')
                        image = image[0]
                        for t, m, s in zip(image, mean, std):
                            t.mul_(s).add_(m)
                        save_results_more(name[0], img_vis_dir, image.cpu().data, \
                                          pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
                                          pred_cnt.item(), gt_count.item())
        print_loss = avg_loss.average() / world_size

        mae = cnt_errors['mae'].avg / world_size
        mse = np.sqrt(cnt_errors['mse'].avg / world_size)
        nae = cnt_errors['nae'].avg / world_size

        if rank == 0:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', print_loss, global_steps)
            writer.add_scalar('valid_mae', mae, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        return print_loss, mae, mse, nae

    def patch_forward(self, config, model, img, dot_map, num_patches, args, mode):
        # crop the img and gt_map with a max stride on x and y axis
        # size: HW: __C_NWPU.TRAIN_SIZE
        # stack them with a the batchsize: __C_NWPU.TRAIN_BATCH_SIZE
        crop_imgs = []
        crop_dots, crop_dens, crop_masks = {}, {}, {}

        den_map = args[0]
        crop_dots['1'], crop_dots['2'], crop_dots['4'], crop_dots['8'] = [], [], [], []
        crop_dens['1'], crop_dens['2'], crop_dens['4'], crop_dens['8'] = [], [], [], []
        crop_masks['1'], crop_masks['2'], crop_masks['4'], crop_masks['8'] = [], [], [], []
        b, c, h, w = img.shape

        crop_size = config.test.get('crop_size', (768, 1024))  # default (768, 1024)
        rh, rw = crop_size

        # support for multi-scale patch forward
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                for res_i in range(len(dot_map)):
                    gis_, gie_ = gis // 2 ** res_i, gie // 2 ** res_i
                    gjs_, gje_ = gjs // 2 ** res_i, gje // 2 ** res_i
                    crop_dots[str(2 ** res_i)].append(dot_map[res_i][:, gis_:gie_, gjs_:gje_])
                    crop_dens[str(2 ** res_i)].append(den_map[res_i][:, :, gis_:gie_, gjs_:gje_])
                    mask = torch.zeros_like(dot_map[res_i]).cpu()
                    mask[:, gis_:gie_, gjs_:gje_].fill_(1.0)
                    crop_masks[str(2 ** res_i)].append(mask)

        crop_imgs = torch.cat(crop_imgs, dim=0)
        for k, v in crop_dots.items():
            crop_dots[k] = torch.cat(v, dim=0)
        for k, v in crop_dens.items():
            crop_dens[k] = torch.cat(v, dim=0)
        for k, v in crop_masks.items():
            crop_masks[k] = torch.cat(v, dim=0)

        # forward may need repeatng
        crop_losses = []
        crop_preds = {}
        crop_labels = {}
        crop_labels['1'], crop_labels['2'], crop_labels['4'], crop_labels['8'] = [], [], [], []
        crop_preds['1'], crop_preds['2'], crop_preds['4'], crop_preds['8'] = [], [], [], []
        nz, bz = crop_imgs.size(0), num_patches
        keys_pre = None

        for i in range(0, nz, bz):
            gs, gt = i, min(nz, i + bz)
            result = model(crop_imgs[gs:gt], [crop_dots[k][gs:gt] for k in crop_dots.keys()],
                           mode, [[crop_dens[k][gs:gt] for k in crop_dens.keys()]])
            crop_pred = result['pre_den']
            crop_label = result['gt_den']

            keys_pre = result['pre_den'].keys()
            for k in keys_pre:
                crop_preds[k].append(crop_pred[k].cpu())
                crop_labels[k].append(crop_label[k].cpu())

            crop_losses.append(result['losses'].mean())

        for k in keys_pre:
            crop_preds[k] = torch.cat(crop_preds[k], dim=0)
            crop_labels[k] = torch.cat(crop_labels[k], dim=0)

        # splice them to the original size

        result = {'pre_den': {}, 'gt_den': {}}

        for res_i, k in enumerate(keys_pre):

            pred_map = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
            labels = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
            idx = 0
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                    gis_, gie_ = gis // 2 ** res_i, gie // 2 ** res_i
                    gjs_, gje_ = gjs // 2 ** res_i, gje // 2 ** res_i

                    pred_map[:, :, gis_:gie_, gjs_:gje_] += crop_preds[k][idx]
                    labels[:, :, gis_:gie_, gjs_:gje_] += crop_labels[k][idx]
                    idx += 1
            # import pdb
            # pdb.set_trace()
            # for the overlapping area, compute average value
            mask = crop_masks[k].sum(dim=0).unsqueeze(0).unsqueeze(0)
            pred_map = (pred_map / mask)
            labels = (labels / mask)
            result['pre_den'].update({k: pred_map})
            result['gt_den'].update({k: labels})
            result.update({'losses': crop_losses[0]})
        return result

class Validator_SMC:

    def validate(self, config, testloader, model, writer_dict, device,
                 num_patches, img_vis_dir, mean, std):

        rank = get_rank()
        world_size = get_world_size()
        model.eval()
        avg_loss = AverageMeter()
        cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(),
                      'nae': AverageMeter(), 'acc1': AverageMeter()}
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                # if _>100:
                #     break
                image, label, _, name = batch
                image = image.to(device)
                for i in range(len(label)):
                    label[i] = label[i].to(device)
                # result = model(image, label, 'val')
                result = self.patch_forward(config, model, image, label, num_patches, 'val')

                losses = result['losses']
                pre_den = result['pre_den']['8']
                gt_den = result['gt_den']['8']

                #    -----------Counting performance------------------
                gt_count, pred_cnt = label[0].sum(), pre_den.sum()

                # print(f" rank: {rank} gt: {gt_count} pre: {pred_cnt}")
                s_mae = torch.abs(gt_count - pred_cnt)

                s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))

                allreduce_tensor(s_mae)
                allreduce_tensor(s_mse)
                # acc1 = reduce_tensor(result['acc1']['error']/(result['acc1']['gt']+1e-10))
                reduced_loss = reduce_tensor(losses)
                # print(f" rank: {rank} mae: {s_mae} mse: {s_mse}"
                #       f"loss: {reduced_loss}")
                avg_loss.update(reduced_loss.item())
                # cnt_errors['acc1'].update(acc1)
                cnt_errors['mae'].update(s_mae.item())
                cnt_errors['mse'].update(s_mse.item())

                s_nae = (torch.abs(gt_count - pred_cnt) / (gt_count + 1e-10))
                allreduce_tensor(s_nae)
                cnt_errors['nae'].update(s_nae.item())

                if rank == 0:
                    if idx % 20 == 0:
                        # acc1 = cnt_errors['acc1'].avg/world_size
                        # print( f'acc1:{acc1}')
                        image = image[0]
                        for t, m, s in zip(image, mean, std):
                            t.mul_(s).add_(m)
                        save_results_more(name[0], img_vis_dir, image.cpu().data, \
                                          pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
                                          pred_cnt.item(), gt_count.item())
        print_loss = avg_loss.average() / world_size

        mae = cnt_errors['mae'].avg / world_size
        mse = np.sqrt(cnt_errors['mse'].avg / world_size)
        nae = cnt_errors['nae'].avg / world_size

        if rank == 0:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', print_loss, global_steps)
            writer.add_scalar('valid_mae', mae, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        return print_loss, mae, mse, nae

    def patch_forward(self, config, model, img, dot_map, num_patches, mode):
        # crop the img and gt_map with a max stride on x and y axis
        # size: HW: __C_NWPU.TRAIN_SIZE
        # stack them with a the batchsize: __C_NWPU.TRAIN_BATCH_SIZE
        crop_imgs = []
        crop_dots, crop_masks = {}, {}

        crop_dots['1'], crop_dots['2'], crop_dots['4'], crop_dots['8'] = [], [], [], []
        crop_masks['1'], crop_masks['2'], crop_masks['4'], crop_masks['8'] = [], [], [], []
        b, c, h, w = img.shape

        crop_size = config.test.get('crop_size', (768, 1024))  # default (768, 1024)
        rh, rw = crop_size

        # support for multi-scale patch forward
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                for res_i in range(len(dot_map)):
                    gis_, gie_ = gis // 2 ** res_i, gie // 2 ** res_i
                    gjs_, gje_ = gjs // 2 ** res_i, gje // 2 ** res_i
                    crop_dots[str(2 ** res_i)].append(dot_map[res_i][:, gis_:gie_, gjs_:gje_])
                    mask = torch.zeros_like(dot_map[res_i]).cpu()
                    mask[:, gis_:gie_, gjs_:gje_].fill_(1.0)
                    crop_masks[str(2 ** res_i)].append(mask)

        crop_imgs = torch.cat(crop_imgs, dim=0)
        for k, v in crop_dots.items():
            crop_dots[k] = torch.cat(v, dim=0)
        for k, v in crop_masks.items():
            crop_masks[k] = torch.cat(v, dim=0)

        # forward may need repeatng
        crop_losses = []
        crop_preds = {}
        crop_labels = {}
        crop_labels['1'], crop_labels['2'], crop_labels['4'], crop_labels['8'] = [], [], [], []
        crop_preds['1'], crop_preds['2'], crop_preds['4'], crop_preds['8'] = [], [], [], []
        nz, bz = crop_imgs.size(0), num_patches
        keys_pre = None

        for i in range(0, nz, bz):
            gs, gt = i, min(nz, i + bz)
            result = model(crop_imgs[gs:gt], [crop_dots[k][gs:gt] for k in crop_dots.keys()],
                           mode)
            crop_pred = result['pre_den']
            crop_label = result['gt_den']

            keys_pre = result['pre_den'].keys()
            for k in keys_pre:
                crop_preds[k].append(crop_pred[k].cpu())
                crop_labels[k].append(crop_label[k].cpu())

            crop_losses.append(result['losses'].mean())

        for k in keys_pre:
            crop_preds[k] = torch.cat(crop_preds[k], dim=0)
            crop_labels[k] = torch.cat(crop_labels[k], dim=0)

        # splice them to the original size

        result = {'pre_den': {}, 'gt_den': {}}

        keys_pre = ['8']
        for res_i, k in enumerate(keys_pre):
            res_i = 3
            pred_map = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
            labels = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
            idx = 0
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                    gis_, gie_ = gis // 2 ** res_i, gie // 2 ** res_i
                    gjs_, gje_ = gjs // 2 ** res_i, gje // 2 ** res_i

                    pred_map[:, :, gis_:gie_, gjs_:gje_] += crop_preds[k][idx]
                    labels[:, :, gis_:gie_, gjs_:gje_] += crop_labels[k][idx]
                    idx += 1
            # import pdb
            # pdb.set_trace()
            # for the overlapping area, compute average value
            mask = crop_masks[k].sum(dim=0).unsqueeze(0).unsqueeze(0)
            pred_map = (pred_map / mask)
            labels = (labels / mask)
            result['pre_den'].update({k: pred_map})
            result['gt_den'].update({k: labels})
            result.update({'losses': crop_losses[0]})
        return result



class Validator_Adaptive_Kernel:

    def validate(self, config, testloader, model, writer_dict, device,
                 num_patches, img_vis_dir, mean, std):

        rank = get_rank()
        world_size = get_world_size()
        model.eval()
        avg_loss = AverageMeter()
        cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(),
                      'nae': AverageMeter(), 'acc1': AverageMeter()}
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                # if _>100:
                #     break
                image, label, _, name, den_map = batch
                image = image.to(device)
                for i in range(len(label)):
                    label[i] = label[i].to(device)
                # result = model(image, label, 'val')
                result = self.patch_forward(config, model, image, label, num_patches, 'val')

                losses = result['losses']
                pre_den = result['pre_den']['1']
                gt_den = result['gt_den']['1']

                #    -----------Counting performance------------------
                gt_count, pred_cnt = label[0].sum(), pre_den.sum()

                # print(f" rank: {rank} gt: {gt_count} pre: {pred_cnt}")
                s_mae = torch.abs(gt_count - pred_cnt)

                s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))

                allreduce_tensor(s_mae)
                allreduce_tensor(s_mse)
                # acc1 = reduce_tensor(result['acc1']['error']/(result['acc1']['gt']+1e-10))
                reduced_loss = reduce_tensor(losses)
                # print(f" rank: {rank} mae: {s_mae} mse: {s_mse}"
                #       f"loss: {reduced_loss}")
                avg_loss.update(reduced_loss.item())
                # cnt_errors['acc1'].update(acc1)
                cnt_errors['mae'].update(s_mae.item())
                cnt_errors['mse'].update(s_mse.item())

                s_nae = (torch.abs(gt_count - pred_cnt) / (gt_count + 1e-10))
                allreduce_tensor(s_nae)
                cnt_errors['nae'].update(s_nae.item())

                if rank == 0:
                    if idx % 20 == 0:
                        # acc1 = cnt_errors['acc1'].avg/world_size
                        # print( f'acc1:{acc1}')
                        image = image[0]
                        for t, m, s in zip(image, mean, std):
                            t.mul_(s).add_(m)
                        save_results_more(name[0], img_vis_dir, image.cpu().data, \
                                          pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
                                          pred_cnt.item(), gt_count.item())
        print_loss = avg_loss.average() / world_size

        mae = cnt_errors['mae'].avg / world_size
        mse = np.sqrt(cnt_errors['mse'].avg / world_size)
        nae = cnt_errors['nae'].avg / world_size

        if rank == 0:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', print_loss, global_steps)
            writer.add_scalar('valid_mae', mae, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        return print_loss, mae, mse, nae

    def patch_forward(self, config, model, img, dot_map, num_patches, mode):
        # crop the img and gt_map with a max stride on x and y axis
        # size: HW: __C_NWPU.TRAIN_SIZE
        # stack them with a the batchsize: __C_NWPU.TRAIN_BATCH_SIZE
        crop_imgs = []
        crop_dots, crop_masks = {}, {}

        crop_dots['1'], crop_dots['2'], crop_dots['4'], crop_dots['8'] = [], [], [], []
        crop_masks['1'], crop_masks['2'], crop_masks['4'], crop_masks['8'] = [], [], [], []
        b, c, h, w = img.shape

        crop_size = config.test.get('crop_size', (768, 1024))  # default (768, 1024)
        rh, rw = crop_size

        # support for multi-scale patch forward
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                for res_i in range(len(dot_map)):
                    gis_, gie_ = gis // 2 ** res_i, gie // 2 ** res_i
                    gjs_, gje_ = gjs // 2 ** res_i, gje // 2 ** res_i
                    crop_dots[str(2 ** res_i)].append(dot_map[res_i][:, gis_:gie_, gjs_:gje_])
                    mask = torch.zeros_like(dot_map[res_i]).cpu()
                    mask[:, gis_:gie_, gjs_:gje_].fill_(1.0)
                    crop_masks[str(2 ** res_i)].append(mask)

        crop_imgs = torch.cat(crop_imgs, dim=0)
        for k, v in crop_dots.items():
            crop_dots[k] = torch.cat(v, dim=0)
        for k, v in crop_masks.items():
            crop_masks[k] = torch.cat(v, dim=0)

        # forward may need repeatng
        crop_losses = []
        crop_preds = {}
        crop_labels = {}
        crop_labels['1'], crop_labels['2'], crop_labels['4'], crop_labels['8'] = [], [], [], []
        crop_preds['1'], crop_preds['2'], crop_preds['4'], crop_preds['8'] = [], [], [], []
        nz, bz = crop_imgs.size(0), num_patches
        keys_pre = None

        for i in range(0, nz, bz):
            gs, gt = i, min(nz, i + bz)
            result = model(crop_imgs[gs:gt], [crop_dots[k][gs:gt] for k in crop_dots.keys()],
                           mode)
            crop_pred = result['pre_den']
            crop_label = result['gt_den']

            keys_pre = result['pre_den'].keys()
            for k in keys_pre:
                crop_preds[k].append(crop_pred[k].cpu())
                crop_labels[k].append(crop_label[k].cpu())

            crop_losses.append(result['losses'].mean())

        for k in keys_pre:
            crop_preds[k] = torch.cat(crop_preds[k], dim=0)
            crop_labels[k] = torch.cat(crop_labels[k], dim=0)

        # splice them to the original size

        result = {'pre_den': {}, 'gt_den': {}}

        for res_i, k in enumerate(keys_pre):

            pred_map = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
            labels = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
            idx = 0
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                    gis_, gie_ = gis // 2 ** res_i, gie // 2 ** res_i
                    gjs_, gje_ = gjs // 2 ** res_i, gje // 2 ** res_i

                    pred_map[:, :, gis_:gie_, gjs_:gje_] += crop_preds[k][idx]
                    labels[:, :, gis_:gie_, gjs_:gje_] += crop_labels[k][idx]
                    idx += 1
            # import pdb
            # pdb.set_trace()
            # for the overlapping area, compute average value
            mask = crop_masks[k].sum(dim=0).unsqueeze(0).unsqueeze(0)
            pred_map = (pred_map / mask)
            labels = (labels / mask)
            result['pre_den'].update({k: pred_map})
            result['gt_den'].update({k: labels})
            result.update({'losses': crop_losses[0]})
        return result