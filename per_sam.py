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
from utils_ import misc
from utils_.eval_metrics import UnsupervisedMetrics
from core.metrics import FewShotMetric
from per_segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator
from timm.models import vit_base_patch32_224_clip_laion2b, vit_base_patch16_224_dino
from data_kits.perseg import PerSeg
import torch.nn.functional as F
import timm

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt
import tqdm

image_encoder = None
transforms = None

ex = setup(
    Experiment(name="FPTrans", save_git_info=False, base_dir="./")
)
torch.set_printoptions(precision=8)


@ex.command(unobserved=True)
def test(_run, _config, exp_id=-1, ckpt=None, strict=True, eval_after_train=False):
    opt, logger, device = init_environment(ex, _run, _config, eval_after_train=eval_after_train)

    torch.set_printoptions(precision=3)

    if opt.dataset in ['PASCAL', 'COCO']:
        ds_test, data_loader, num_classes = datasets.load(opt, logger, "test", transform='whatever')
        ds_test.reset_sampler()
        ds_test.sample_tasks()
    else:
        ds_test = PerSeg()
        num_classes = len(ds_test.paths)

    logger.info(f'     ==> {len(ds_test)} testing samples')

    sam_checkpoint = "data/pretrained/sam/sam_vit_h_4b8939.pth"

    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    global image_encoder, transforms
    image_encoder = timm.create_model('vit_giant_patch14_dinov2.lvd142m', pretrained=True)
    image_encoder = image_encoder.eval().cuda()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(image_encoder)
    transforms = Compose([
        Resize(size=(518, 518), interpolation=Image.BICUBIC, max_size=None),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    mask_predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    results = []
    bar = tqdm.tqdm(ds_test)
    metric_m1 = FewShotMetric(n_class=num_classes)
    metric_m2 = FewShotMetric(n_class=num_classes)
    metric_m3 = FewShotMetric(n_class=num_classes)
    npoints = 1
    if opt.dataset in ['COCO']:
        np.random.seed(42)
        test_sample_indices = np.random.choice(np.arange(len(ds_test)), 500)
    else:
        test_sample_indices = np.arange(len(ds_test))
    bar = tqdm.tqdm(test_sample_indices)

    for ii in bar:
        sample = ds_test[ii]

        if opt.dataset in ['PASCAL', 'COCO']:
            image = sample['qry_rgb'].permute(1, 2, 0).numpy() * 255
            target = sample['qry_msk']
            ref_image = sample['sup_rgb'][0].permute(1, 2, 0).numpy() * 255
            ref_mask = sample['sup_msk'][0:1].permute(1, 2, 0).numpy()
            image = image.astype(np.uint8)
            ref_image = ref_image.astype(np.uint8)
            sample_cls = sample['cls']
            if ref_mask is not None:
                ref_mask[ref_mask > 10] = 0
        else:
            image = sample['qry_rgb']
            target = sample['qry_msk'][:, :, 0]
            target = target > 0
            ref_image = sample['sup_rgb']
            ref_mask = sample['sup_msk'][:, :, 0:1]
            sample_cls = sample['cls']

        mask, sim_image, (masks, sims) = semantic_parts(mask_generator, predictor=mask_predictor,
                              test_image=image, ref_image=ref_image, ref_mask=ref_mask, encoder=image_encoder)
        # mask = persam(mask_predictor, ref_mask, ref_image, target, image, 1)
        # mask = persam_f(mask_predictor, ref_mask, ref_image, target, image, 1)
        # masks = mask_generator.generate(image)

        # masks, point_coords, point_labels = persam(
        #     mask_predictor,
        #     ref_mask,
        #     ref_image,
        #     target,
        #     image,
        #     k=1000,
        #     image_encoder=clip_image_encoder
        # )

        metric_m1.update(mask[None, ...], target[None, ...], [sample['cls']], verbose=False)
        # metric_m2.update((masks[1] > 0)[None, ...], target[None, ...], [sample['cls']], verbose=False)
        # metric_m3.update((masks[2] > 0)[None, ...], target[None, ...], [sample['cls']], verbose=False)
        if opt.dataset in ['PASCAL', 'COCO']:
            miou_class_1, miou_avg_1 = metric_m1.get_scores(datasets.get_val_labels(opt, None))
        else:
            miou_class_1, miou_avg_1 = metric_m1.get_scores(list(range(1, num_classes)))
        # miou_class_2, miou_avg_2 = metric_m1.get_scores(datasets.get_val_labels(opt, None))
        # miou_class_3, miou_avg_3 = metric_m1.get_scores(datasets.get_val_labels(opt, None))
        bar.set_description(f'miou: {miou_avg_1}')
        print(miou_class_1)

        plt.figure(figsize=(20, 20))
        plt.subplot(3, 1, 1)
        plt.imshow(image)
        # show_mask(mask, ax=plt, random_color=True)
        show_masks(plt, masks, sims)
        # show_anns(masks)
        plt.axis('off')
        plt.subplot(3, 1, 2)
        plt.imshow(image)
        show_mask(target > 0, ax=plt, random_color=True)
        # show_points(point_coords, labels=point_labels, ax=plt)
        plt.axis('off')
        plt.subplot(3, 1, 3)
        plt.imshow(sim_image)
        # plt.imshow(ref_image)

        # show_mask(ref_mask[:, :, 0], ax=plt, random_color=True)
        plt.axis('off')
        # for ii, mask in enumerate(masks):
        #     plt.subplot(4, 1, ii + 2)
        #     plt.imshow(image)
        #     show_mask(mask, ax=plt, random_color=True)
        #     # show_points(point_coords, labels=point_labels, ax=plt)
        #     plt.axis('off')
        plt.savefig('result.png')
        plt.close()

    return f"mIoU: {np.mean(results) * 100:.2f}"

def show_masks(ax, masks, sims):
    for mask, sim in zip(masks, sims):
        mask = mask['segmentation']
        sim = (sim - sims.min()) / (sims.max() - sims.min())
        color = np.concatenate([np.array([0, 0, 1.0]) * sim.item(), np.array([1.0])], axis=0)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


@torch.no_grad()
def semantic_parts(generator: SamAutomaticMaskGenerator,
                   predictor: SamPredictor, test_image, ref_image, ref_mask, encoder=None):
    shape = test_image.shape[0:2]
    masks = generator.generate(test_image)

    clip_image_encoder = encoder

    if ref_mask is not None:
        ref_mask[ref_mask > 10] = 0

    # ref_mask, ref_image = predictor.set_image(ref_image, ref_mask)
    # ref_feat = predictor.features.squeeze().permute(1, 2, 0)
    # ref_image = F.interpolate(ref_image, size=(224, 224), mode='bilinear')
    # ref_feat = clip_image_encoder.forward_features(ref_image)
    # ref_feat = ref_feat[:, 1:, ...].reshape(14, 14, 768)
    # # print(ref_feat.shape)
    # clip_ref_feat = clip_embedding(ref_mask, ref_image, clip_image_encoder, predictor)

    mask_transforms = Compose(transforms.transforms[0:-1])
    ref_image = transforms(Image.fromarray(ref_image))
    if ref_mask is not None:
        ref_mask = mask_transforms(Image.fromarray(ref_mask.squeeze())).cuda()
    ref_feat = image_encoder.forward_features(ref_image.unsqueeze(0).cuda()).squeeze()[1:].reshape(37, 37, -1)

    # ref_mask = torch.as_tensor(ref_mask[:, :0], device=ref_feat.device)
    # ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="nearest")
    # ref_mask = ref_mask.squeeze()
    # print(ref_mask.shape)

    target_embedding = pooled_embedding(ref_mask, predictor, ref_feat)

    # _, test_image = predictor.set_image(test_image)
    # test_feat = predictor.features.squeeze().permute(1, 2, 0)
    # test_image = F.interpolate(test_image, size=(224, 224), mode='bilinear')
    # test_feat = clip_image_encoder.forward_features(test_image)
    # test_feat = test_feat[:, 1:, ...].reshape(14, 14, 768)
    test_image = transforms(Image.fromarray(test_image))
    test_feat = image_encoder.forward_features(test_image.unsqueeze(0).cuda()).squeeze()[1:].reshape(37, 37, -1)

    mask_embeddings = []
    for mask_dict in masks:
        mask = mask_dict['segmentation']
        # mask = predictor.transform.apply_mask(mask.astype(np.int32))
        # mask = torch.as_tensor(mask, device=test_feat.device)
        # mask = predictor.model.preprocess_mask(mask)

        mask = mask_transforms(Image.fromarray(mask))
        mask = torch.as_tensor(mask, device=ref_feat.device)
        # mask = predictor.model.preprocess_mask(mask)

        mask_embedding = pooled_embedding(mask, predictor, test_feat)

        # mask_embedding = pooled_embedding(mask, predictor, test_feat)
        # clip_mask_embedding = clip_embedding(mask, test_image, clip_image_encoder, predictor)
        # mask_embedding = mask_embedding / mask_embedding.norm(dim=-1, keepdim=True)
        # mask_embeddings.append(clip_mask_embedding)
        mask_embeddings.append(mask_embedding)

    # print(test_feat.shape, target_embedding.shape)
    image_sim = test_feat / test_feat.norm(dim=-1, keepdim=True) \
                @ (target_embedding / target_embedding.norm(dim=-1, keepdim=True)).t()

    image_sim = image_sim.squeeze()

    image_sim = F.interpolate(image_sim[None, None, ...], size=shape, mode='bilinear')
    # image_sim = image_sim.squeeze()

    # image_sim = predictor.model.postprocess_masks(
    #             image_sim[None, None, ...],
    #             input_size=predictor.input_size,
    #             original_size=predictor.original_size)
    # print(torch.histogram(image_sim.cpu()))

    embeddings = torch.cat(mask_embeddings, dim=0)
    print(embeddings.shape, ref_feat.shape)
    sim = F.cosine_similarity(embeddings, target_embedding, dim=1)
    sim = torch.nan_to_num(sim, -1)
    best_mask = sim.flatten().topk(5)[1]
    print(sim.flatten()[best_mask])

    mask = np.zeros_like(masks[0]['segmentation'])
    for i in best_mask:
        if sim.flatten()[i] > 0.85:
            mask = np.logical_or(masks[i]['segmentation'], mask)

    return mask, image_sim.squeeze().cpu(), [masks, sim.flatten()]


@torch.no_grad()
def pooled_embedding(mask, predictor, test_feat):
    mask = mask.squeeze()
    test_feat = F.interpolate(test_feat.permute(2, 0, 1).unsqueeze(0), size=mask.shape[0:2], mode='bilinear')
    # mask = mask.squeeze()
    test_feat = test_feat.squeeze().permute(1, 2, 0)
    mask_feat = test_feat[mask > 0]
    target_feat_mean = mask_feat.mean(0)
    if len(mask_feat) == 0:
        return target_feat_mean.unsqueeze(0)
    target_feat_max = torch.max(mask_feat, dim=0)[0]
    target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

    return target_feat


@torch.no_grad()
def clip_embedding(mask, image, encoder, predictor):

    masked_image = image.clone()
    masked_image[mask.repeat(1, 3, 1, 1) < 1] = 0
    masked_image = F.interpolate(masked_image, size=(224, 224), mode='bilinear')
    return encoder(masked_image)

def persam_f(predictor: SamPredictor, ref_mask, ref_image, test_mask, test_image, k, image_encoder=None):
    for name, param in predictor.model.named_parameters():
        param.requires_grad = False


    gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0
    gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()
    # Image features encoding
    ref_mask, _ = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask.float(), size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()

    # Target feature extraction
    target_feat = ref_feat[ref_mask > 0]

    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)

    # Cosine similarity
    h, w, C = ref_feat.shape
    target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
    sim = target_feat @ ref_feat

    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
                    sim,
                    input_size=predictor.input_size,
                    original_size=predictor.original_size).squeeze()

    # Positive location prior
    # topk_xy, topk_label = point_selection(sim, topk=1)
    topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
    topk_xy = np.concatenate([topk_xy_i], axis=0)
    topk_label = np.concatenate([topk_label_i], axis=0)

    print('======> Start Training')
    # Learnable mask weights
    mask_weights = Mask_Weights().cuda()
    mask_weights.train()
    lr = 1e-3
    train_epoch = 1000
    log_epoch = 200

    optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epoch)

    for train_idx in range(train_epoch):

        # Run the decoder
        masks, scores, logits, logits_high = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=True)
        logits_high = logits_high.flatten(1)

        # Weighted sum three-scale masks
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        logits_high = logits_high * weights
        logits_high = logits_high.sum(0).unsqueeze(0)

        dice_loss = calculate_dice_loss(logits_high, gt_mask)
        focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
        loss = dice_loss + focal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if train_idx % log_epoch == 0:
            print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(),
                                                                             focal_loss.item()))

    mask_weights.eval()
    weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
    weights_np = weights.detach().cpu().numpy()
    print('======> Mask weights:\n', weights_np)

    print('======> Start Testing')
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
    topk_xy = np.concatenate([topk_xy_i], axis=0)
    topk_label = np.concatenate([topk_label_i], axis=0)

    sim = (sim - sim.mean()) / torch.std(sim)
    sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

    # First-step prediction
    masks, scores, logits, logits_high = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        multimask_output=True)

    # Weighted sum three-scale masks
    logits_high = logits_high * weights.unsqueeze(-1)
    logit_high = logits_high.sum(0)
    mask = (logit_high > 0).detach().cpu().numpy()

    logits = logits * weights_np[..., None]
    logit = logits.sum(0)

    # Cascaded Post-refinement-1
    y, x = np.nonzero(mask)
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        box=input_box[None, :],
        mask_input=logit[None, :, :],
        multimask_output=True)
    best_idx = np.argmax(scores)

    # Cascaded Post-refinement-2
    y, x = np.nonzero(masks[best_idx])
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        box=input_box[None, :],
        mask_input=logits[best_idx: best_idx + 1, :, :],
        multimask_output=True)
    best_idx = np.argmax(scores)

    return masks[best_idx]

import torch.nn as nn
class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

@torch.no_grad()
def persam(predictor: SamPredictor, ref_mask, ref_image, test_mask, test_image, k, image_encoder=None):
    # Image features encoding
    ref_mask, _ = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()

    # Target feature extraction
    target_feat = ref_feat[ref_mask > 0]

    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)

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

    sim = (sim - sim.mean()) / torch.std(sim)
    sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

    # First-step prediction
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        multimask_output=False,
        attn_sim=attn_sim,
        target_embedding=target_embedding
    )
    best_idx = 0

    # Cascaded Post-refinement-1
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        mask_input=logits[best_idx: best_idx + 1, :, :],
        multimask_output=True)
    best_idx = np.argmax(scores)

    # Cascaded Post-refinement-2
    y, x = np.nonzero(masks[best_idx])
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, scores, logits, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        box=input_box[None, :],
        mask_input=logits[best_idx: best_idx + 1, :, :],
        multimask_output=True)
    best_idx = np.argmax(scores)

    return masks[best_idx]


def point_selection(mask_sim, topk=1, random=False):
    # Top-1 point selection
    if random == True:
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten(0).topk(100)[1]
        topk_xy = topk_xy[np.random.randint(topk_xy.shape[0], size=5)]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * 5)
        topk_xy = topk_xy.cpu().numpy()

    else:
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
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
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
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


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

if __name__ == '__main__':
    ex.run_commandline()
