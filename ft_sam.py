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
from networks.mask_classifier import MaskClassifier
from utils_ import misc
from utils_.eval_metrics import UnsupervisedMetrics
from core.metrics import FewShotMetric
from per_segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator
from timm.models import vit_base_patch32_224_clip_laion2b, vit_base_patch16_224_dino
from torchmetrics import JaccardIndex
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import tqdm

from per_sam import show_masks, show_anns, show_points, pooled_embedding, show_mask
from data_kits.perseg import PerSeg

ex = setup(
    Experiment(name="FPTrans", save_git_info=False, base_dir="./")
)
torch.set_printoptions(precision=8)


@ex.command(unobserved=True)
def test(_run, _config, exp_id=-1, ckpt=None, strict=True, eval_after_train=False):
    opt, logger, device = init_environment(ex, _run, _config, eval_after_train=eval_after_train)

    dataset = 'perseg'
    if dataset == 'voc':
        ds_test, data_loader, num_classes = datasets.load(opt, logger, "test", transform='whatever')
        ds_test.reset_sampler()
        ds_test.sample_tasks()
    else:
        ds_test = PerSeg()
        num_classes = len(ds_test.paths)
    logger.info(f'     ==> {len(ds_test)} testing samples')

    sam_checkpoint = "data/pretrained/sam/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    classifier = MaskClassifier(dim=256, num_heads=4).cuda()

    mask_predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=32)
    results = []
    bar = tqdm.tqdm(range(len(ds_test)))
    metric_m1 = FewShotMetric(n_class=num_classes)

    for ii in bar:
        if ii < 4: continue
        sample = ds_test[ii]
        if dataset == 'voc':
            image = sample['qry_rgb'].permute(1, 2, 0).numpy() * 255
            target = sample['qry_msk']
            ref_image = sample['sup_rgb'][0].permute(1, 2, 0).numpy() * 255
            ref_mask = sample['sup_msk'][0:1].permute(1, 2, 0).numpy()
            image = image.astype(np.uint8)
            ref_image = ref_image.astype(np.uint8)
            sample_cls = sample['cls']
        else:
            image = sample['qry_rgb']
            target = sample['qry_msk'][:, :, 0]
            target = target > 0
            ref_image = sample['sup_rgb']
            ref_mask = sample['sup_msk'][:, :, 0:1]
            sample_cls = sample['cls']

        masks, labels, mask = semantic_parts(mask_generator, predictor=mask_predictor, classifier=classifier,
                                       test_image=image, ref_image=ref_image, ref_mask=ref_mask)

        metric_m1.update(mask[None, ...], target[None, ...], [sample['cls']], verbose=False)
        miou_class_1, miou_avg_1 = metric_m1.get_scores(list(range(1, num_classes)))
        bar.set_description(f"mIoU 1: {miou_avg_1} class_miou: {miou_class_1}")

        #
        plt.figure(figsize=(20, 20))
        plt.subplot(4, 1, 1)
        plt.imshow(image)
        show_mask(mask, ax=plt, random_color=True)
        # show_anns(masks)
        plt.axis('off')
        plt.subplot(4, 1, 2)
        plt.imshow(image)
        show_mask(target > 0, ax=plt, random_color=True)
        # show_points(point_coords, labels=point_labels, ax=plt)
        plt.axis('off')
        plt.subplot(4, 1, 3)
        # plt.imshow(sim_image)
        plt.imshow(ref_image)

        # show_mask(ref_mask[:, :, 0], ax=plt, random_color=True)
        show_masks(plt, masks, labels.float())
        plt.axis('off')
        # for ii, mask in enumerate(masks):
        #     plt.subplot(4, 1, ii + 2)
        #     plt.imshow(image)
        #     show_mask(mask, ax=plt, random_color=True)
        #     # show_points(point_coords, labels=point_labels, ax=plt)
        #     plt.axis('off')
        plt.axis('off')
        plt.subplot(4, 1, 4)
        # plt.imshow(sim_image)
        plt.imshow(image)
        # show_masks(plt, masks, labels)
        plt.axis('off')
        plt.savefig('result.png')
        plt.close()

    return f"mIoU: {np.mean(results) * 100:.2f}"

def post_refinement():
    pass


@torch.no_grad()
def semantic_parts(generator, predictor, classifier, test_image, ref_image, ref_mask):
    del classifier
    classifier = MaskClassifier(256, 4, output_size=2).cuda()

    ref_sam_masks, ref_embeddings, ref_img_labels, gt_embedding = \
        semantic_parts_embedding(generator, predictor, ref_image, ref_mask)

    test_sam_masks, test_embeddings, _, _ = \
        semantic_parts_embedding(generator, predictor, test_image, None)

    ref_input = torch.cat((gt_embedding, ref_embeddings), dim=0)
    train_mask_classifier(classifier, ref_input, ref_img_labels)

    test_input = torch.cat((gt_embedding, ref_embeddings, test_embeddings), dim=0)
    sim = torch.argmax(classifier(test_input[None, ...]), dim=-1)
    sim = sim.squeeze()[1+ref_embeddings.shape[0]:]

    # num_masks = 15 if len(sim) > 15 else len(sim)
    # best_mask = sim.flatten().topk(num_masks)[1]
    # print(sim.flatten()[best_mask])
    #
    mask = np.zeros_like(test_sam_masks[0]['segmentation']).astype(float)
    for i in range(len(sim.flatten())):
        if sim.flatten()[i] > 0.001:
            mask = np.logical_or(test_sam_masks[i]['segmentation'], mask)
        # mask += test_sam_masks[i]['segmentation'].astype(float) * sim.flatten()[i].item()
        # mask[test_sam_masks[i]['segmentation']] = sim.flatten()[i].item()

    # # Cascaded Post-refinement-
    # y, x = np.nonzero(mask > 0)
    # x_min = x.min()
    # x_max = x.max()
    # y_min = y.min()
    # y_max = y.max()
    #
    # mask = predictor.transform.apply_mask(mask.astype(np.int32))
    # mask = torch.as_tensor(mask, device=ref_input.device)
    # mask = predictor.model.preprocess_mask(mask)
    # mask = F.interpolate(mask[None, None, ...].float(), size=(256, 256), mode='nearest').cpu().numpy()
    # mask = mask.squeeze()
    #
    # input_box = np.array([x_min, y_min, x_max, y_max])
    # masks, scores, logits, _ = predictor.predict(
    #     box=input_box[None, :],
    #     mask_input=mask[None, ...],
    #     multimask_output=True)
    # best_idx = np.argmax(scores)
    # print(logits.shape)

    # Cascaded Post-refinement-2
    # y, x = np.nonzero(masks[best_idx])
    # x_min = x.min()
    # x_max = x.max()
    # y_min = y.min()
    # y_max = y.max()
    # input_box = np.array([x_min, y_min, x_max, y_max])
    # masks, scores, logits, _ = predictor.predict(
    #     box=input_box[None, :],
    #     mask_input=logits[best_idx: best_idx + 1, :, :],
    #     multimask_output=True)
    # best_idx = np.argmax(scores)
    # print(best_idx)


    return ref_sam_masks, ref_img_labels, mask


def train_mask_classifier(classifier, ref_embeddings: torch.Tensor, ref_img_labels:torch.Tensor):
    with torch.enable_grad():
        ref_iters = 200
        ref_lr = 0.0001
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(classifier.parameters(), lr=ref_lr, eps=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, ref_iters)

        for ii in range(ref_iters):
            ref_pred = classifier.forward(ref_embeddings[None, ...])
            loss = criterion(ref_pred.squeeze()[1:], ref_img_labels.long())
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


def semantic_parts_embedding(generator, predictor, ref_image, ref_mask):
    # trivial solution: intersection over union and training with soft label
    masks = generator.generate(ref_image)

    ref_mask, ref_image = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    target_embedding = None
    if ref_mask is not None:
        ref_mask[ref_mask == 255] = 0
        target_embedding = pooled_embedding(ref_mask, predictor, ref_feat)

    metric = JaccardIndex(task="binary").to(ref_feat.device)

    mask_embeddings = []

    labels = []
    masks_ret = []
    for mask_dict in masks:
        mask = mask_dict['segmentation']
        mask = predictor.transform.apply_mask(mask.astype(np.int32))
        mask = torch.as_tensor(mask, device=ref_feat.device)
        mask = predictor.model.preprocess_mask(mask)

        mask_embedding = pooled_embedding(mask, predictor, ref_feat)

        if torch.isnan(mask_embedding).any():
            continue

        if ref_mask is not None:
            iou = metric(mask.squeeze(), ref_mask.squeeze().bool())
            if torch.isnan(iou):
                continue
            labels.append(iou > 0.01)
        mask_embeddings.append(mask_embedding)
        masks_ret.append(mask_dict)

    mask_embeddings_ret = torch.cat(mask_embeddings, dim=0)
    labels_ret = None
    if ref_mask is not None:
        labels_ret = torch.tensor(labels, device=mask_embeddings_ret.device)

    return masks_ret, mask_embeddings_ret, labels_ret, target_embedding


def classify_semantic_parts(classifier, mask_embeddings, support_embedding):
    # L, N, E with batch at dim=1
    embeddings = torch.cat((mask_embeddings, support_embedding), dim=0)
    prediction = classifier(embeddings)

    return prediction


if __name__ == '__main__':
    ex.run_commandline()
