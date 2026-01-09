import SimpleITK as sitk
import numpy as np
import os,csv,json
from surface_distance import metrics

# src_root =  '/home/meng/ailab/data/CMR/3d_segmentation/img_to_mask/3d_pc_unet/val4/prediction_results/'
# src_root = '/home/meng/ailab/data/CMR/3d_segmentation/img_to_mask/2d_unet/val4/prediction_results/'

# src_root = '/home/meng/ailab/data/CMR/3d_segmentation/img_to_mask/2d_unet/val4/prediction_results/'

# src_root =  '/home/meng/ailab/data/2018Atrium_seg/3d_segmentation_la/img_to_mask/3d_pc_unet/val4/prediction_results'

# src_root = '/home/meng/ailab/data/2018Atrium_seg/3d_segmentation_la/img_to_mask/3d_pc_unet/val4_cat/prediction_results/'

src_root = '/home/meng/ailab/data/2018Atrium_seg/3d_segmentation_la/img_to_mask/3d_u_net/val4/prediction_results/'



th_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

evaluate_results = {}
evaluate_results['max_dice'] = 0
evaluate_results['max_dice_threshold'] = 0

file_num = 4

for th_value in th_list:
    evaluate_results['threshold_' + str(th_value)] = {}
    total_dice = 0
    patients = 0
    for subroot, dirs, files in os.walk(src_root):
        if len(files) < file_num: continue
        pre_mask_file = 'mask_prediction.nii.gz'
        patient_num_vec = subroot.split('/')
        p1 = patient_num_vec[-1]
        print('working on %s', p1)
        mask_file = os.path.join(subroot, pre_mask_file)
        mask_image = sitk.ReadImage(mask_file)

        size = mask_image.GetSize() #x, y, z
        np_mask_img = sitk.GetArrayFromImage(mask_image)
        np_lv_volume = np.zeros(size)
        lv_mask_value = th_value

        lv_index = np.where(np_mask_img >= lv_mask_value ) #z, y, x
        lv_index_z = lv_index[0]
        lv_index_y = lv_index[1]
        lv_index_x = lv_index[2]

        slice_num = size[2]
        np_lv_volume = np.zeros((slice_num, size[1], size[0]))
        for i in range(len(lv_index[0])): np_lv_volume[lv_index_z[i], lv_index_y[i], lv_index_x[i]] = 1
        pre_np_mask = np.uint8(np_lv_volume)

        gt_mask = 'mask_gt.nii.gz'
        gt_mask_file = os.path.join(subroot, gt_mask)
        gt_mask_image = sitk.ReadImage(gt_mask_file)
        gt_np_mask = np.uint8(sitk.GetArrayFromImage(gt_mask_image))

        # dice = metrics.my_dice_loss(pred=pre_np_mask, target=gt_np_mask )
        dice = metrics.compute_dice_coefficient(mask_gt=gt_np_mask, mask_pred=pre_np_mask)
        evaluate_results['threshold_' + str(th_value)][p1] = {}
        evaluate_results['threshold_' + str(th_value)][p1]['Dice'] = dice
        total_dice += dice
        patients += 1

    avg_dice = total_dice / patients
    evaluate_results['threshold_' + str(th_value)]['average_dice'] = avg_dice

    if avg_dice > evaluate_results['max_dice']:
        evaluate_results['max_dice'] = avg_dice
        evaluate_results['max_dice_threshold'] = th_value

target_root = src_root
with open(os.path.join(target_root, 'eval_thresold_results.json'), 'w', encoding='utf-8') as f_json:
    json.dump(evaluate_results, f_json, indent=4)


for subroot, dirs, files in os.walk(src_root):
    if len(files) < file_num: continue
    pre_mask_file = 'mask_prediction.nii.gz'
    patient_num = pre_mask_file
    print('working on %s', patient_num)

    patient_num_vec = subroot.split('/')
    p1 = patient_num_vec[-3]
    mask_file = os.path.join(subroot, pre_mask_file)
    mask_image = sitk.ReadImage(mask_file)

    origin = mask_image.GetOrigin()
    direction = mask_image.GetDirection()
    spacing = mask_image.GetSpacing()
    size = mask_image.GetSize() #x, y, z

    np_mask_img = sitk.GetArrayFromImage(mask_image)
    np_lv_volume = np.zeros(size)
    lv_mask_value = evaluate_results['max_dice_threshold']

    lv_index = np.where(np_mask_img >= lv_mask_value ) #z, y, x
    lv_index_z = lv_index[0]
    lv_index_y = lv_index[1]
    lv_index_x = lv_index[2]

    slice_num = size[2]
    np_lv_volume = np.zeros((slice_num, size[1], size[0]))
    for i in range(len(lv_index[0])): np_lv_volume[lv_index_z[i], lv_index_y[i], lv_index_x[i]] = 1

    lv_volume = sitk.GetImageFromArray(np_lv_volume)
    lv_volume.SetOrigin(origin)
    lv_volume.SetDirection(direction)
    lv_volume.SetSpacing(spacing)

    temp_root = subroot


    sitk.WriteImage(lv_volume, os.path.join(temp_root, 'mask_prediction_th.nii.gz'))










