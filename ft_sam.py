from pathlib import Path

import os
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
from torchvision.ops import focal_loss
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import matplotlib.pyplot as plt
import tqdm
import timm

from per_sam import show_masks, show_anns, show_points, pooled_embedding, show_mask, point_selection
from data_kits.perseg import PerSeg

ex = setup(
    Experiment(name="FPTrans", save_git_info=False, base_dir="./")
)
torch.set_printoptions(precision=8)
image_encoder = None
transforms = None


@ex.command(unobserved=True)
def test(
        _run,
        _config,
        exp_id=-1,
        ckpt=None,
        strict=True,
        eval_after_train=False,
        epochs=1000,
        lr=0.0001,
        threshold=0.01,
        dataset='perseg',
        topk=3
):
    opt, logger, device = init_environment(ex, _run, _config, eval_after_train=eval_after_train)

    if dataset in ['PASCAL', 'COCO']:
        ds_test, data_loader, num_classes = datasets.load(opt, logger, "test", transform='whatever')
        ds_test.reset_sampler()
        ds_test.sample_tasks()
    else:
        ds_test = PerSeg()
        num_classes = len(ds_test.paths)
    logger.info(f'     ==> {len(ds_test)} testing samples')

    sam_checkpoint = "data/pretrained/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    global image_encoder, transforms
    image_encoder = timm.create_model('vit_giant_patch14_dinov2.lvd142m', pretrained=True)
    image_encoder = image_encoder.eval().cuda()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(image_encoder)
    # transforms = timm.data.create_transform(**data_config, is_training=False)
    transforms = Compose([
        Resize(size=(518, 518), interpolation=Image.BICUBIC, max_size=None),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    classifier = MaskClassifier(1536, 4, output_size=1).cuda()

    mask_predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32, points_per_batch=64)
    results = []

    seed = 42
    np.random.seed(seed)
    if dataset in ['COCO']:
        test_sample_indices = np.random.choice(np.arange(len(ds_test)), 500)
    else:
        test_sample_indices = np.arange(len(ds_test))
    bar = tqdm.tqdm(test_sample_indices)
    metric_m1 = FewShotMetric(n_class=num_classes)

    plt.imshow(ds_test[test_sample_indices[2]]['qry_rgb'].permute(1, 2, 0))
    plt.savefig('test.png')
    exit()

    for index, ii in enumerate(bar):
        sample = ds_test[ii]
        if dataset in ['PASCAL', 'COCO']:
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

        masks, labels, mask, box = semantic_parts(
            mask_generator,
            predictor=mask_predictor,
            classifier=classifier,
            test_image=image,
            ref_image=ref_image,
            ref_mask=ref_mask,
            epochs=epochs,
            lr=lr,
            threshold=threshold,
            topk=topk
        )

        metric_m1.update(mask[None, ...], target[None, ...], [sample['cls']], verbose=False)
        if dataset in ['PASCAL', 'COCO']:
            miou_class_1, miou_avg_1 = metric_m1.get_scores(datasets.get_val_labels(opt, None))
        else:
            miou_class_1, miou_avg_1 = metric_m1.get_scores(list(range(1, num_classes)))
        bar.set_description(f"mIoU 1: {miou_avg_1}")
        print(miou_class_1)

        #
        plt.figure(figsize=(20, 20))
        ax = plt.subplot(4, 1, 1)
        ax.imshow(image)
        show_mask(mask, ax=ax, random_color=True)
        # show_box(box, ax=ax)
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
        plt.imshow(ref_image)
        # show_masks(plt, masks, labels)
        show_mask(ref_mask.squeeze(), ax=plt, random_color=True)
        plt.axis('off')
        os.makedirs(f'images/perseg/{exp_id}/', exist_ok=True)
        plt.savefig(f'images/perseg/{exp_id}/{index:04d}.png')
        plt.close()

    return f"mIoU: {np.mean(results) * 100:.2f}"

def post_refinement():
    pass


@torch.no_grad()
def semantic_parts(
        generator,
        predictor,
        classifier,
        test_image,
        ref_image,
        ref_mask,
        epochs=1000,
        threshold=0.01,
        lr=1e-4,
        topk=3,
):
    classifier = MaskClassifier(1536, 4, output_size=1).cuda()

    best_mask, ref_img_labels, ref_input, \
    ref_sam_masks, sim, test_sam_masks = mask_generation(classifier, epochs,
                                                         generator, lr, predictor,
                                                         ref_image, ref_mask,
                                                         test_image, topk)
    print(sim.flatten()[best_mask])
    #
    mask = np.zeros_like(test_sam_masks[0]['segmentation']).astype(float)

    points = []
    point_labels = []
    boxes = []
    for i in best_mask:
        if sim.flatten()[i].item() > threshold:
            mask = np.logical_or(test_sam_masks[i]['segmentation'], mask)
        # mask += test_sam_masks[i]['segmentation'].astype(float) * sim.flatten()[i].item()
        # mask[test_sam_masks[i]['segmentation']] = sim.flatten()[i].item()

    #     topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(torch.as_tensor(test_sam_masks[i]['segmentation']).float(), topk=1, random=True)
    #     points.append(topk_xy_i)
    #     point_labels.append(topk_label_i)
    #
    #         # # Cascaded Post-refinement-
    # y, x = np.nonzero(mask > 0)
    # x_min = x.min()
    # x_max = x.max()
    # y_min = y.min()
    # y_max = y.max()
    # print(points, point_labels)
    #
    # input_box = np.array([x_min, y_min, x_max, y_max])
    #
    # mask = predictor.transform.apply_mask(mask.astype(np.int32))
    # mask = torch.as_tensor(mask, device=ref_input.device)
    # mask = predictor.model.preprocess_mask(mask)
    # mask = F.interpolate(mask[None, None, ...].float(), size=(256, 256), mode='nearest').cpu().numpy()
    # mask = mask.squeeze()
    #
    # predictor.set_image(test_image)
    #
    # masks, scores, logits, _ = predictor.predict(
    #     point_coords=np.concatenate(points, axis=0),
    #     point_labels=np.concatenate(point_labels, axis=0),
    #     mask_input=mask[None, ...],
    #     multimask_output=True)
    # best_idx = np.argmax(scores)
    #
    # # Cascaded Post-refinement-2
    # y, x = np.nonzero(masks[best_idx])
    # x_min = x.min()
    # x_max = x.max()
    # y_min = y.min()
    # y_max = y.max()
    # input_box = np.array([x_min, y_min, x_max, y_max])
    # masks, scores, logits, _ = predictor.predict(
    #     point_coords=np.concatenate(points, axis=0),
    #     point_labels=np.concatenate(point_labels, axis=0),
    #     box=input_box[None, ...],
    #     mask_input=logits[best_idx: best_idx + 1, :, :],
    #     multimask_output=True)
    # best_idx = np.argmax(scores)
    # print(best_idx)

    return ref_sam_masks, ref_img_labels, mask, None


def mask_generation(classifier, epochs, generator, lr, predictor, ref_image, ref_mask, test_image, topk):
    ref_sam_masks, ref_embeddings, ref_img_labels, gt_embedding = \
        semantic_parts_embedding(generator, predictor, ref_image, ref_mask)
    test_sam_masks, test_embeddings, _, _ = \
        semantic_parts_embedding(generator, predictor, test_image, None)
    print(gt_embedding.shape, ref_embeddings.shape)
    ref_input_ref = torch.cat((gt_embedding, ref_embeddings), dim=0).unsqueeze(0)
    ref_input = torch.cat(
        (ref_embeddings.squeeze().unsqueeze(1), ref_input_ref.repeat(ref_embeddings.shape[0], 1, 1)),
        dim=1)
    train_mask_classifier(classifier, ref_input, ref_img_labels, ref_mask, ref_sam_masks, epochs=epochs, lr=lr)

    # test_input_ref = torch.cat((gt_embedding, ref_embeddings, test_embeddings), dim=0)
    test_input = torch.cat(
        (test_embeddings.squeeze().unsqueeze(1), ref_input_ref.repeat(test_embeddings.shape[0], 1, 1)),
        dim=1)
    # test_input = torch.cat((gt_embedding, test_embeddings), dim=0)
    sim = classifier(test_input)
    sim = torch.sigmoid(sim)
    sim = sim.squeeze()[:, 0]
    # sim = sim.squeeze()[1:]
    num_masks = topk if len(sim) > topk else len(sim)
    best_mask = sim.flatten().topk(num_masks)[1]
    return best_mask, ref_img_labels, ref_input, ref_sam_masks, sim, test_sam_masks


def train_mask_classifier(
        classifier,
        ref_embeddings: torch.Tensor,
        ref_img_labels: torch.Tensor,
        ref_mask,
        ref_sam_masks,
        epochs=1000,
        lr=1e-4
):
    with torch.enable_grad():
        ref_iters = epochs
        ref_lr = lr
        criterion = focal_loss.sigmoid_focal_loss
        optimizer = optim.AdamW(classifier.parameters(), lr=ref_lr, eps=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, ref_iters)

        bar = range(ref_iters)
        for ii in bar:
            ref_pred = classifier.forward(ref_embeddings)
            loss = criterion(ref_pred.squeeze()[:, 0], ref_img_labels.float(), reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

def calculate_dice_loss(inputs, targets, num_masks = 1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def semantic_parts_embedding(generator, predictor, ref_image, ref_mask):
    # trivial solution: intersection over union and training with soft label
    masks = generator.generate(ref_image)

    # ref_mask, ref_image = predictor.set_image(ref_image, ref_mask)
    # ref_feat = predictor.features.squeeze().permute(1, 2, 0)
    mask_transforms = Compose(transforms.transforms[0:-1])
    ref_image = transforms(Image.fromarray(ref_image))
    if ref_mask is not None:
        ref_mask = mask_transforms(Image.fromarray(ref_mask.squeeze())).cuda()
    ref_feat = image_encoder.forward_features(ref_image.unsqueeze(0).cuda()).squeeze()[1:].reshape(37, 37, -1)

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
        # mask = predictor.transform.apply_mask(mask.astype(np.int32))
        mask = mask_transforms(Image.fromarray(mask))
        mask = torch.as_tensor(mask, device=ref_feat.device)
        # mask = predictor.model.preprocess_mask(mask)

        mask_embedding = pooled_embedding(mask, predictor, ref_feat)

        if torch.isnan(mask_embedding).any():
            continue

        if ref_mask is not None:
            iou = metric(mask.squeeze(), ref_mask.squeeze().bool())
            if torch.isnan(iou):
                continue
            labels.append(iou > 0.05)
        mask_embeddings.append(mask_embedding)
        masks_ret.append(mask_dict)

    mask_embeddings_ret = torch.cat(mask_embeddings, dim=0)
    labels_ret = None
    if ref_mask is not None:
        labels_ret = torch.tensor(labels, device=mask_embeddings_ret.device)

    print(mask_embeddings_ret.shape)
    return masks_ret, mask_embeddings_ret, labels_ret, target_embedding


def classify_semantic_parts(classifier, mask_embeddings, support_embedding):
    # L, N, E with batch at dim=1
    embeddings = torch.cat((mask_embeddings, support_embedding), dim=0)
    prediction = classifier(embeddings)

    return prediction

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

if __name__ == '__main__':
    ex.run_commandline()
