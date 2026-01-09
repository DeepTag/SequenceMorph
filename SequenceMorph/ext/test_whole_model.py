import sys
sys.path.append("..")

import torch
from ME_nets.LagrangianMotionEstimationNet import Lagrangian_motion_estimate_net, Lagrangian_motion_residual_refinement_net, miccai2018_net_cc_san_grid_warp
from data_set.load_data_for_cine_ME import add_np_data, get_np_data_as_groupids,load_np_datagroups, DataType, load_Dataset
from surface_distance import metrics
import os, json, csv, shutil
import SimpleITK as sitk
import numpy as np
import time
# import matplotlib.pyplot as plt


# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_dec_weights(model, weights):
    print('Resuming net weights from {} ...'.format(weights))
    w_dict = torch.load(weights)
    model.load_state_dict(w_dict, strict=True)
    return model

def test_Cardiac_Tagging_ME_net(net1, \
                                net2, \
                                np_data_root, \
                                val_dataset_files, \
                                model_name1, \
                                model_name2, \
                                model_path, \
                                dst_root, \
                                case = 'proposed'):
    with open(val_dataset_files, 'r', encoding='utf-8') as f_json:
        data_config = json.load(f_json)
    if data_config is not None and data_config.__class__ is dict:
        grouped_data_sets = data_config.get('validation')
        if grouped_data_sets.__class__ is not dict: print('invalid train_config.')

    # check grouped_data_sets
    if grouped_data_sets.__class__ is not dict: print('invalid data config file.')

    group_names = grouped_data_sets.keys()
    val_data_list = []
    for group_name in group_names:
        print('working on %s', group_name)
        filesListDict = grouped_data_sets.get(group_name)
        if filesListDict.__class__ is not dict: continue

        for sample in filesListDict.keys():
            each_trainingSets = filesListDict.get(sample)
            # list images_data_niix in each dataset
            cine_npz = each_trainingSets.get('cine')
            val_data_list.append(cine_npz)

    validation_data_group_ids = get_np_data_as_groupids(model_root=np_data_root, data_type=DataType.VALIDATION)
    validation_cines, validation_tags  = load_np_datagroups(np_data_root, validation_data_group_ids,
                                                                           data_type=DataType.VALIDATION)
    val_dataset = load_Dataset(validation_cines, validation_tags)
    val_batch_size = 1
    test_set_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
    net1_model = os.path.join(model_path, model_name1)
    ME_model1 = load_dec_weights(net1, net1_model)
    ME_model1 = ME_model1.to(device)
    ME_model1.eval()

    net2_model = os.path.join(model_path, model_name2)
    ME_model = load_dec_weights(net2, net2_model)
    ME_model = ME_model.to(device)
    ME_model.eval()
    time_list = []

    for i, data in enumerate(test_set_loader):
        # cine0, tag = data
        untagged_cine, tagged_cine = data

        # wrap input data in a Variable object
        cine1 = tagged_cine.to(device)

        # wrap input data in a Variable object
        img = cine1.cuda()
        img = img.float()

        x = img[:, 1:, ::]  # other frames except the 1st frame
        y = img[:, 0:24, ::]  # 1st frame also is the reference frame
        shape = x.shape  # batch_size, seq_length, height, width
        seq_length = shape[1]
        height = shape[2]
        width = shape[3]
        x = x.contiguous()
        x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        # y = y.repeat(1, seq_length, 1, 1)  # repeat the ES frame to match other frames contained in a Cine
        y = y.contiguous()
        y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width


        # forward pass
        training_start_time = time.time()
        with torch.no_grad():
            val_registered_cine1, val_registered_cine2, val_registered_cine1_lag1, \
            val_flow_param1, val_deformation_matrix, val_deformation_matrix_neg, val_deformation_matrix_lag = net1(y, x)

            val_registered_cine1_lag2, lag_val_registered_cine2, val_registered_cine1_lag1_refined, val_flow_param2, \
            val_deformation_matrix2, val_deformation_matrix_neg2, refined_val_deformation_matrix_lag = net2(val_registered_cine1_lag1, x, val_deformation_matrix_lag, img[:, 0:1, ::])
        infer_time = time.time() - training_start_time
        time_list.append(infer_time)

        y = img[:, -1, ::]  # the last frame

        val_deformation_matrix_lag0 = torch.cat((refined_val_deformation_matrix_lag[:,0,::], y), dim=0)
        val_deformation_matrix_lag0 = val_deformation_matrix_lag0.cuda()
        val_deformation_matrix_lag0 = val_deformation_matrix_lag0.cpu().detach().numpy()
        # val_deformation_matrix_lag0 = val_deformation_matrix_lag0.squeeze(0)

        val_deformation_matrix_lag1 = torch.cat((refined_val_deformation_matrix_lag[:, 1, ::], y), dim=0)
        val_deformation_matrix_lag1 = val_deformation_matrix_lag1.cuda()
        val_deformation_matrix_lag1 = val_deformation_matrix_lag1.cpu().detach().numpy()
        # val_deformation_matrix_lag1 = val_deformation_matrix_lag1.squeeze(0)

        y0 = img[:, 0, ::]  # the last frame

        val_registered_cine = torch.cat((y0, val_registered_cine1.squeeze(1)), dim=0)
        val_registered_cine = val_registered_cine.cuda()
        val_registered_cine = val_registered_cine.cpu().detach().numpy()

        val_registered_cine_lag = torch.cat((y0, val_registered_cine1_lag1.squeeze(1)), dim=0)
        val_registered_cine_lag = val_registered_cine_lag.cuda()
        val_registered_cine_lag = val_registered_cine_lag.cpu().detach().numpy()

        val_registered_cine2 = torch.cat((y0, val_registered_cine1_lag2.squeeze(1)), dim=0)
        val_registered_cine2 = val_registered_cine2.cuda()
        val_registered_cine2 = val_registered_cine2.cpu().detach().numpy()

        val_registered_cine_lag2 = torch.cat((y0, val_registered_cine1_lag1_refined.squeeze(1)), dim=0)
        val_registered_cine_lag2 = val_registered_cine_lag2.cuda()
        val_registered_cine_lag2 = val_registered_cine_lag2.cpu().detach().numpy()


        file_path = val_data_list[i][0]
        root_vec = file_path.split(os.path.sep)
        tgt_root1 = os.path.join(dst_root, root_vec[-5])
        if not os.path.exists(tgt_root1): os.mkdir(tgt_root1)
        tgt_root2 = os.path.join(tgt_root1, root_vec[-3])
        if not os.path.exists(tgt_root2): os.mkdir(tgt_root2)
        tgt_root3 = os.path.join(tgt_root2, root_vec[-2])
        if not os.path.exists(tgt_root3): os.mkdir(tgt_root3)

        root = '/ailab/data/cardiac_ME/training/'
        val_img_file_root = os.path.join(root, root_vec[-3], root_vec[-2])

        val_cine_file = 'cine.nii.gz'

        cine_image = sitk.ReadImage(os.path.join(val_img_file_root, val_cine_file))
        spacing1 = cine_image.GetSpacing()
        origin1 = cine_image.GetOrigin()
        direction1 = cine_image.GetDirection()

        img_matrix = sitk.GetArrayFromImage(cine_image)
        cine_img = sitk.GetImageFromArray(img_matrix)
        cine_img.SetSpacing(spacing1)
        cine_img.SetDirection(direction1)
        cine_img.SetOrigin(origin1)
        sitk.WriteImage(cine_img, os.path.join(tgt_root3, val_cine_file))


        val_deformation_matrix_lag_img0 = sitk.GetImageFromArray(val_deformation_matrix_lag0)
        val_deformation_matrix_lag_img0.SetSpacing(spacing1)
        val_deformation_matrix_lag_img0.SetOrigin(origin1)
        val_deformation_matrix_lag_img0.SetDirection(direction1)
        sitk.WriteImage(val_deformation_matrix_lag_img0, os.path.join(tgt_root3, 'deformation_matrix_x.nii.gz'))

        val_deformation_matrix_lag_img1 = sitk.GetImageFromArray(val_deformation_matrix_lag1)
        val_deformation_matrix_lag_img1.SetSpacing(spacing1)
        val_deformation_matrix_lag_img1.SetOrigin(origin1)
        val_deformation_matrix_lag_img1.SetDirection(direction1)
        sitk.WriteImage(val_deformation_matrix_lag_img1, os.path.join(tgt_root3, 'deformation_matrix_y.nii.gz'))

        val_registered_cine_img = sitk.GetImageFromArray(val_registered_cine)
        val_registered_cine_img.SetSpacing(spacing1)
        val_registered_cine_img.SetOrigin(origin1)
        val_registered_cine_img.SetDirection(direction1)
        sitk.WriteImage(val_registered_cine_img, os.path.join(tgt_root3, 'f_registered_cine_img.nii.gz'))

        val_registered_cine_lag_img = sitk.GetImageFromArray(val_registered_cine_lag)
        val_registered_cine_lag_img.SetSpacing(spacing1)
        val_registered_cine_lag_img.SetOrigin(origin1)
        val_registered_cine_lag_img.SetDirection(direction1)
        sitk.WriteImage(val_registered_cine_lag_img, os.path.join(tgt_root3, 'lag_registered_cine_img.nii.gz'))

        val_registered_cine_img = sitk.GetImageFromArray(val_registered_cine2)
        val_registered_cine_img.SetSpacing(spacing1)
        val_registered_cine_img.SetOrigin(origin1)
        val_registered_cine_img.SetDirection(direction1)
        sitk.WriteImage(val_registered_cine_img, os.path.join(tgt_root3, 'f_registered_cine_img_refine.nii.gz'))

        val_registered_cine_lag_img = sitk.GetImageFromArray(val_registered_cine_lag2)
        val_registered_cine_lag_img.SetSpacing(spacing1)
        val_registered_cine_lag_img.SetOrigin(origin1)
        val_registered_cine_lag_img.SetDirection(direction1)
        sitk.WriteImage(val_registered_cine_lag_img, os.path.join(tgt_root3, 'lag_registered_cine_img_refine.nii.gz'))

        print('finish: ' + str(i))
    print(np.mean(time_list, axis=0))
    print(np.std(time_list, axis=0, ddof=1))
    return time_list


def test_Cardiac_cine_ME_net_mask(net, grid_root, flow_root, dst_root):
    if not os.path.exists(dst_root): os.makedirs(dst_root)
    for subroot, dirs, files in os.walk(flow_root):
        if len(files) < 3: continue
        root_vec = subroot.split(os.path.sep)
        # tgt_root1 = os.path.join(dst_root, root_vec[-5])
        # if not os.path.exists(tgt_root1): os.mkdir(tgt_root1)
        tgt_root2 = os.path.join(dst_root, root_vec[-3])
        if not os.path.exists(tgt_root2): os.mkdir(tgt_root2)
        tgt_root20 = os.path.join(tgt_root2, root_vec[-2])
        if not os.path.exists(tgt_root20): os.mkdir(tgt_root20)
        tgt_root3 = os.path.join(tgt_root20, root_vec[-1])
        if not os.path.exists(tgt_root3): os.mkdir(tgt_root3)


        for file in files:
            if file.endswith('.nii.gz') and 'deformation_matrix_x' in file:
                deformation_matrix_x_img_file = file
            if file.endswith('.nii.gz') and 'deformation_matrix_y' in file:
                deformation_matrix_y_img_file = file
                # break

        my_source_file = os.path.join(subroot, 'cine.nii.gz')
        my_target_file = os.path.join(tgt_root3, 'cine.nii.gz')
        shutil.copy(my_source_file, my_target_file)
        my_source_file = os.path.join(subroot, 'lag_registered_cine_img.nii.gz')
        my_target_file = os.path.join(tgt_root3, 'lag_registered_cine_img.nii.gz')
        shutil.copy(my_source_file, my_target_file)
        my_source_file = os.path.join(subroot, 'lag_registered_cine_img_refine.nii.gz')
        my_target_file = os.path.join(tgt_root3, 'lag_registered_cine_img_refine.nii.gz')
        shutil.copy(my_source_file, my_target_file)

        deformation_matrix_x_img = sitk.ReadImage(os.path.join(subroot, deformation_matrix_x_img_file))
        deformation_matrix_y_img = sitk.ReadImage(os.path.join(subroot, deformation_matrix_y_img_file))
        deformation_matrix_x = sitk.GetArrayFromImage(deformation_matrix_x_img)
        deformation_matrix_y = sitk.GetArrayFromImage(deformation_matrix_y_img)

        val_mask_file_root = os.path.join(grid_root, root_vec[-3], root_vec[-2], root_vec[-1])

        lm3_file = False

        for subroot, dirs, files in os.walk(val_mask_file_root):
            if len(files) < 3: continue
            for file in files:
                if file.endswith('nii.gz') and 'ED' in file:
                    ED_mask_file = os.path.join(subroot, file)
                    sitk_ED_mask_image = sitk.ReadImage(ED_mask_file)
                    # ED_num_vec = ED_mask_file.split('.')
                    # ED_num = int(ED_num_vec[1])
                    lm3_file = True
                    sitk.WriteImage(sitk_ED_mask_image, os.path.join(tgt_root3, file))
                if file.endswith('nii.gz') and 'ES' in file:
                    ES_mask_file = os.path.join(subroot, file)
                    # ES_num_vec = ES_mask_file.split('.')
                    # ES_num = int(ES_num_vec[1])
                    gt_ES_mask_image = sitk.ReadImage(ES_mask_file)
                    sitk.WriteImage(gt_ES_mask_image, os.path.join(tgt_root3, file))

        if not lm3_file: continue
        # if ED_num != 1: print('ED frame is not 1')
        gt_ED_mask_image = sitk.GetArrayFromImage(sitk_ED_mask_image)


        # wrap input data in a Variable object
        flow_x = torch.from_numpy(deformation_matrix_x).to(device)
        flow_y = torch.from_numpy(deformation_matrix_y).to(device)
        flow_x = flow_x.float()
        flow_y = flow_y.float()


        shape = flow_x.shape  #seq_length, height, width
        seq_length = shape[0]
        height = shape[1]
        width = shape[2]
        x = flow_x.contiguous()
        x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width


        y = flow_y.contiguous()
        y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        # wrap input data in a Variable object
        tag_image = torch.from_numpy(gt_ED_mask_image.astype(float)).to(device)
        # wrap input data in a Variable object
        tag_image = tag_image.float()
        z = tag_image

        z = z.contiguous()
        z0 = z.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width
        z0 = (z0[0, ::]).view(-1, 1, height, width)


        eular_flow = torch.cat((x, y), dim=1) # DeepTag-python
        grids_eular = torch.zeros(shape)
        grids_eular = grids_eular.to(device)
        grids_eular[0,::] = z0

        for lm in range(1, 4):
            ones = lm*torch.ones_like(z0)
            zeros = torch.zeros_like(z0)
            z_lm = torch.where(z0 == lm, ones, z0)
            z_lm = torch.where(z_lm != lm, zeros, z_lm)
            if torch.sum(z_lm) > 0.1:
                for i in range (0, shape[0]-1):
                    flow = eular_flow[i, ::]
                    flow = flow.unsqueeze(0)
                    grid = net(z_lm, flow)
                    grid = torch.where(grid >= 0.2*lm, ones, grid)
                    grid = torch.where(grid < 0.2 * lm, zeros, grid)
                    grid = grid.squeeze(0)
                    grid = grid.squeeze(0)
                    grids_eular[i+1, ::] += grid
                    new_grid = grids_eular[i+1, ::]
                    new_grid = torch.where(new_grid > lm, ones, new_grid)
                    new_grid = new_grid.squeeze(0)
                    new_grid = new_grid.squeeze(0)
                    grids_eular[i + 1, ::] = new_grid

            # """


        grids = grids_eular
        resgistered_grids = grids.cpu().detach().numpy()

        spacing1 = deformation_matrix_x_img.GetSpacing()
        origin1 = deformation_matrix_x_img.GetOrigin()
        direction1 = deformation_matrix_x_img.GetDirection()

        resgistered_grids_img = sitk.GetImageFromArray(resgistered_grids)
        resgistered_grids_img.SetSpacing(spacing1)
        resgistered_grids_img.SetOrigin(origin1)
        resgistered_grids_img.SetDirection(direction1)
        sitk.WriteImage(resgistered_grids_img, os.path.join(tgt_root3, 'warped_mask.nii.gz'))

        spacing2 = gt_ES_mask_image.GetSpacing()
        origin2 = gt_ES_mask_image.GetOrigin()
        direction2 = gt_ES_mask_image.GetDirection()

        resgistered_grids_img = sitk.GetImageFromArray(resgistered_grids[-1:, ::])
        resgistered_grids_img.SetSpacing(spacing2)
        resgistered_grids_img.SetOrigin(origin2)
        resgistered_grids_img.SetDirection(direction2)
        sitk.WriteImage(resgistered_grids_img, os.path.join(tgt_root3, 'ED2ES.nii.gz'))

        print('finish: ' + str(0))


if __name__ == '__main__':
    # data loader

    test_dataset = '/ailab/data/cardiac_ME/val1_reverse/configs/Cardiac_ME_test_config.json'
    np_data_root = '/ailab/data/cardiac_ME/val1_reverse/np_data/'

    if not os.path.exists(np_data_root):
        os.mkdir(np_data_root)
        add_np_data(project_data_config_files=test_dataset, data_type='validation', model_root=np_data_root)

    # proposed model
    vol_size = (256, 256)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 16, 3]
    ME_net1 = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec, test_phase=True)
    ME_net2 = Lagrangian_motion_residual_refinement_net(vol_size, nf_enc, nf_dec, test_phase=True)

    net2 = miccai2018_net_cc_san_grid_warp(vol_size)
    net2 = net2.to(device)

    test_model_path = '/ailab/models/cardiac_ME/SequenceMorph'

    test_model = 'SequenceMorph_NCC'
    dst_root_all_models = os.path.join(test_model_path, test_model)
    if not os.path.exists(dst_root_all_models): os.mkdir(dst_root_all_models)
    print(dst_root_all_models)
    test_model_path222 = os.path.join(dst_root_all_models, 'test_stats_results_all_models')
    if not os.path.exists(test_model_path222): os.mkdir(test_model_path222)
    model_name1 = '/Stage1/S1_model.pth'
    model_name2 = '/Stage2/S2_model.pth'
    dice_dic = {}
    test_n = 0

    for subroot, dirs, files in os.walk(test_model_path):
        if len(files) < 2: continue
        for file in files:
            # if test_n > 2: break
            if file.endswith('.pth') and 'model' in file:
                model_name = file
                epoch_vector = model_name2.split('_')
                epoch = epoch_vector[0]
                test_n += 1
                test_time = test_Cardiac_Tagging_ME_net(net1=ME_net1,
                                            net2=ME_net2,
                                         np_data_root=np_data_root,
                                         val_dataset_files=test_dataset,
                                         model_path= test_model_path,
                                         model_name1=model_name1,
                                         model_name2=model_name2,
                                         dst_root=dst_root_all_models,
                                         case = 'proposed')
                # """
                lm_root = '/ailab/data/cardiac_ME/'
                flow_root = dst_root_all_models
                dst_root_masks = os.path.join(test_model_path, test_model + '_masks/')
                if not os.path.exists(dst_root_masks): os.mkdir(dst_root_masks)
                # """
                test_Cardiac_cine_ME_net_mask(net=net2, grid_root=lm_root, flow_root=flow_root, dst_root=dst_root_masks)
                # """
                # evaluation
                src_root = dst_root_masks

                dice_dic[epoch] = []
                dice_dic[epoch].append(int(epoch))
                avg_total_dice = 0
                avg_total_HD = 0
                avg_total_dice_vec = []
                avg_total_HD_vec = []

                for lm in range(1, 4):
                    evaluate_results = {}
                    avg_dice = 0
                    avg_HD = 0
                    dice_vec = []
                    HD_vec = []
                    patients = 0
                    for subroot, dirs, files in os.walk(os.path.join(src_root, 'training')):
                        if len(files) < 6: continue
                        patient_num_vec = subroot.split('/')
                        p1 = patient_num_vec[-2]
                        s1 = patient_num_vec[-1]
                        print('working on %s', p1)

                        for file in files:
                            if file.endswith('.nii.gz') and 'ES' in file:
                                if 'ED' in file:
                                    pre_mask = file
                                else:
                                    gt_mask = file

                        gt_mask_file = os.path.join(subroot, gt_mask)
                        gt_mask_image = sitk.ReadImage(gt_mask_file)
                        gt_np_mask = np.uint8(sitk.GetArrayFromImage(gt_mask_image))

                        pre_mask_file = os.path.join(subroot, pre_mask)
                        pre_mask_image = sitk.ReadImage(pre_mask_file)
                        pre_np_mask = np.uint8(sitk.GetArrayFromImage(pre_mask_image))

                        # dice = metrics.my_dice_loss(pred=pre_np_mask, target=gt_np_mask )
                        ones = lm * np.ones(gt_np_mask.shape)
                        zeros = np.zeros(gt_np_mask.shape)
                        mask_gt_lm = np.where(gt_np_mask == lm, ones, gt_np_mask)
                        mask_gt_lm = np.where(mask_gt_lm != lm, zeros, mask_gt_lm)
                        if np.sum(mask_gt_lm) < 0.5: continue  # skip the slice without the cardiac anatomy structure
                        pre_np_mask_lm = np.where(pre_np_mask == lm, ones, pre_np_mask)
                        pre_np_mask_lm = np.where(pre_np_mask_lm != lm, zeros, pre_np_mask_lm)
                        if np.sum(
                            pre_np_mask_lm) < 0.5: continue  # skip the slice without the cardiac anatomy structure
                        mask_gt_lm = np.uint8(mask_gt_lm)
                        pre_np_mask_lm = np.uint8(pre_np_mask_lm)
                        spacing = pre_mask_image.GetSpacing()

                        dice = metrics.compute_dice_coefficient(mask_gt=mask_gt_lm, mask_pred=pre_np_mask_lm)
                        surface_distances = metrics.compute_surface_distances(
                            mask_gt=mask_gt_lm,
                            mask_pred=pre_np_mask_lm, spacing_mm=(spacing[2], spacing[1], spacing[0]))
                        H_distance = metrics.compute_robust_hausdorff(surface_distances=surface_distances, percent=95)

                        evaluate_results[p1 + '_' + s1] = []
                        evaluate_results[p1 + '_' + s1].append(p1 + '_' + s1)
                        evaluate_results[p1 + '_' + s1].append(round(dice, 3))
                        evaluate_results[p1 + '_' + s1].append(round(H_distance, 3))
                        dice_vec.append(dice)
                        HD_vec.append(H_distance)

                        patients += 1
                        avg_dice += dice
                        avg_HD += H_distance

                    evaluate_results['avg'] = []
                    evaluate_results['avg'].append('Average')
                    evaluate_results['avg'].append(round(avg_dice / patients, 3))
                    evaluate_results['avg'].append(round(avg_HD / patients, 3))
                    evaluate_results['avg2'] = []
                    evaluate_results['avg2'].append('Average2')
                    evaluate_results['avg2'].append(np.mean(dice_vec))
                    evaluate_results['avg2'].append(np.mean(HD_vec))
                    evaluate_results['std2'] = []
                    evaluate_results['std2'].append('Std2')
                    evaluate_results['std2'].append(np.std(dice_vec, ddof=1))
                    evaluate_results['std2'].append(np.std(HD_vec, ddof=1))

                    dst_out_path = os.path.join(test_model_path222,
                                                'test_results_all_' + epoch + '_' + str(lm) + '_' + str(round(avg_dice / patients, 3)))
                    if not os.path.exists(dst_out_path): os.mkdir(dst_out_path)

                    with open(os.path.join(dst_out_path, 'cmr_seg_info_' + str(lm) + '.csv'), 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for key in evaluate_results.keys():
                            writer.writerow(evaluate_results.get(key))

                    avg_total_dice_vec.append(np.mean(dice_vec))
                    avg_total_HD_vec.append(np.mean(HD_vec))
                    dice_dic[epoch].append(np.mean(dice_vec))
                    dice_dic[epoch].append(np.std(dice_vec, ddof=1))
                    dice_dic[epoch].append(np.mean(HD_vec))
                    dice_dic[epoch].append(np.std(HD_vec, ddof=1))

                dice_dic[epoch].append(np.mean(avg_total_dice_vec))
                dice_dic[epoch].append(np.std(avg_total_dice_vec, ddof=1))
                dice_dic[epoch].append(np.mean(avg_total_HD_vec))
                dice_dic[epoch].append(np.std(avg_total_HD_vec, ddof=1))
                dice_dic[epoch].append(np.mean(test_time, axis=0))
                dice_dic[epoch].append(np.std(test_time, axis=0, ddof=1))

    csv_target_root = os.path.join(test_model_path222, 'Dice_all_models_results.csv')
    with open(csv_target_root, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key in dice_dic.keys():
            writer.writerow(dice_dic.get(key))









