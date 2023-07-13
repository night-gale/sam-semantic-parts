from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sacred import Experiment

from config import setup, init_environment
from constants import on_cloud
from core.base_trainer import BaseTrainer, BaseEvaluator
from core.losses import get as get_loss_obj
from data_kits import datasets
from networks import load_model
from utils_ import misc
from utils_.eval_metrics import UnsupervisedMetrics
from core.metrics import FewShotMetric
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch.nn.functional as F

import matplotlib.pyplot as plt
import tqdm

ex = setup(
    Experiment(name="FPTrans", save_git_info=False, base_dir="./")
)
torch.set_printoptions(precision=8)


@ex.command(unobserved=True)
def test(_run, _config, exp_id=-1, ckpt=None, strict=True, eval_after_train=False):
    opt, logger, device = init_environment(ex, _run, _config, eval_after_train=eval_after_train)

    ds_test, data_loader, num_classes = datasets.load(opt, logger, "test", transform='whatever')
    ds_test.reset_sampler()
    ds_test.sample_tasks()
    logger.info(f'     ==> {len(ds_test)} testing samples')

    sam_checkpoint = "data/pretrained/sam/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_predictor = SamPredictor(sam)
    # mask_generator = SamAutomaticMaskGenerator(sam)
    results = []
    bar = tqdm.tqdm(ds_test)
    metric_m1 = FewShotMetric(n_class=num_classes)
    metric_m2 = FewShotMetric(n_class=num_classes)
    metric_m3 = FewShotMetric(n_class=num_classes)
    npoints = 1

    for sample in bar:
        image = sample['qry_rgb'].permute(1, 2, 0).numpy() * 255
        target = sample['qry_msk']
        image = image.astype(np.uint8)
        # mask = mask_generator.generate(image)
        mask_predictor.set_image(image)
        pos = (target == 1).nonzero()
        point_index = np.random.randint(0, len(pos), npoints)

        point_coords = np.stack((pos[point_index, 2], pos[point_index, 1]), axis=1)
        point_labels = np.array([1]*npoints)

        masks, scores, logits = mask_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels
        )

        # sorted_anns = sorted(mask, key=(lambda x: x['area']), reverse=True)
        # mask_list = []
        # for ann in sorted_anns:
        #     m = ann['segmentation']
        #     mask_list.append(m)
        # pred = np.stack(mask_list, axis=0)
        #
        # metric = UnsupervisedMetrics(
        #     'sam',
        #     n_classes=2,
        #     extra_clusters=pred.shape[0] - 2,
        #     compute_hungarian=True,
        # )
        # target = target.int()
        #
        # metric.update(torch.argmax(torch.from_numpy(pred.astype(np.int32)), dim=0), target)
        # result = metric.compute()
        # print(metric.assignments_t)
        # results.append(result['sammIoU'])
        # bar.set_description(f"mIoU: {np.mean(results)}")
        metric_m1.update((masks[0] > 0)[None, ...], target[None, ...], [sample['cls']], verbose=False)
        metric_m2.update((masks[1] > 0)[None, ...], target[None, ...], [sample['cls']], verbose=False)
        metric_m3.update((masks[2] > 0)[None, ...], target[None, ...], [sample['cls']], verbose=False)
        miou_class_1, miou_avg_1 = metric_m1.get_scores(datasets.get_val_labels(opt, None))
        miou_class_2, miou_avg_2 = metric_m1.get_scores(datasets.get_val_labels(opt, None))
        miou_class_3, miou_avg_3 = metric_m1.get_scores(datasets.get_val_labels(opt, None))
        bar.set_description(f"mIoU 1: {miou_avg_1} mIoU 2: {miou_avg_2} mIoU 3: {miou_avg_3} ")

        plt.figure(figsize=(20, 20))
        plt.subplot(4, 1, 1)
        plt.imshow(image)
        show_mask(target > 0, ax=plt, random_color=True)
        show_points(point_coords, labels=point_labels, ax=plt)
        plt.axis('off')
        for ii, mask in enumerate(masks):
            plt.subplot(4, 1, ii + 2)
            plt.imshow(image)
            show_mask(mask, ax=plt, random_color=True)
            show_points(point_coords, labels=point_labels, ax=plt)
            plt.axis('off')
        plt.savefig('result.png')
        plt.close()
        input()

    return f"mIoU: {np.mean(results) * 100:.2f}"


def persam(predictor: SamPredictor, ref_mask, ref_image, test_mask, test_image):
    # Image features encoding
    ref_mask = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    # Target feature extraction
    target_feat = ref_feat[ref_mask > 0]
    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)

    predictor.set_image(test_image)
    test_feat = predictor.features.squeeze()

    # Cosine similarity
    C, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    test_feat = test_feat.reshape(C, h * w)
    sim = target_feat @ test_feat

    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
        sim,
        input_size=predictor.input_size,
        original_size=predictor.original_size).squeeze()

    # Positive-negative location prior
    topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
    topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        multimask_output=False,
    )


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()

    return topk_xy, topk_label, last_xy, last_label

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

@ex.command(unobserved=True)
def predict(_run, _config, exp_id=-1, ckpt=None, strict=True):
    opt, logger, device = init_environment(ex, _run, _config)

    model = load_model(opt, logger)
    if not opt.no_resume:
        model_ckpt = misc.find_snapshot(_run.run_dir.parent, exp_id, ckpt)
        logger.info(f"     ==> Try to load checkpoint from {model_ckpt}")
        model.load_weights(model_ckpt, logger, strict)
        logger.info(f"     ==> Checkpoint loaded.")
    model = model.to(device)
    loss_obj = get_loss_obj(opt, logger, loss='ce')

    sup_rgb, sup_msk, qry_rgb, qry_msk, qry_ori = datasets.load_p(opt, device)
    classes = torch.LongTensor([opt.p.cls]).cuda()

    logger.info("Start predicting.")

    model.eval()
    ret_values = []
    for i in range(qry_rgb.shape[0]):
        print('Processing:', i + 1)
        qry_rgb_i = qry_rgb[i:i + 1]
        qry_msk_i = qry_msk[i:i + 1] if qry_msk is not None else None
        qry_ori_i = qry_ori[i]

        output = model(qry_rgb_i, sup_rgb, sup_msk, out_shape=qry_ori_i.shape[-3:-1])
        pred = output['out'].argmax(dim=1).detach().cpu().numpy()

        if qry_msk_i is not None:
            loss = loss_obj(output['out'], qry_msk_i).item()
            ref = qry_msk_i.cpu().numpy()
            tp = int((np.logical_and(pred == 1, ref != 255) * np.logical_and(ref == 1, ref != 255)).sum())
            fp = int((np.logical_and(pred == 1, ref != 255) * np.logical_and(ref != 1, ref != 255)).sum())
            fn = int((np.logical_and(pred != 1, ref != 255) * np.logical_and(ref == 1, ref != 255)).sum())
            mean_iou = tp / (tp + fp + fn)
            binary_iou = 0
            ret_values.append(f"Loss: {loss:.4f}, mIoU: {mean_iou * 100:.2f}, bIoU: {binary_iou * 100:.2f}")
        else:
            ret_values.append(None)

        # Save to file
        if opt.p.out:
            pred = pred[0].astype(np.uint8) * 255
            if opt.p.overlap:
                out = qry_ori_i.copy()
                out[pred == 255] = out[pred == 255] * 0.5 + np.array([255, 0, 0]) * 0.5
            else:
                out = pred

            out_dir = Path(opt.p.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = Path(opt.p.qry or opt.p.qry_rgb[i]).stem + '_pred.png'
            out_path = out_dir / out_name
            Image.fromarray(out).save(out_path)

        # Release memory
        del output
        torch.cuda.empty_cache()

    if ret_values[0] is not None:
        return '\n'.join(ret_values)


if __name__ == '__main__':
    ex.run_commandline()
