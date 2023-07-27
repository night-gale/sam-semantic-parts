import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from argparse import Namespace

from SegGPT_inference.seggpt_engine import inference_image, inference_video, run_one_image
from data_kits import datasets
from SegGPT_inference import models_seggpt
from core.metrics import FewShotMetric

from sacred import Experiment
from config import setup, init_environment

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


ex = setup(
    Experiment(name="FPTrans", save_git_info=False, base_dir="SegGPT_inference/")
)
torch.set_printoptions(precision=8)


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='SegGPT_inference/seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types',
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./')
    return parser.parse_args([])


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


@ex.command(unobserved=True)
def main(_run, _config, exp_id=-1, ckpt=None, strict=True, eval_after_train=False):
    opt, logger, device = init_environment(ex, _run, _config, eval_after_train=eval_after_train)
    args = get_args_parser()
    print(args)

    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
    print('Model loaded.')

    ds_test, data_loader, num_classes = datasets.load(opt, logger, "test", transform='whatever')
    ds_test.reset_sampler()
    ds_test.sample_tasks()

    bar = tqdm.tqdm(range(len(ds_test)))
    metric_m1 = FewShotMetric(n_class=num_classes)

    if opt.dataset in ['COCO']:
        np.random.seed(42)
        test_sample_indices = np.random.choice(np.arange(len(ds_test)), 500)
    else:
        test_sample_indices = np.arange(len(ds_test))
    bar = tqdm.tqdm(test_sample_indices)

    for ii in bar:
        sample = ds_test[ii]

        image = sample['qry_rgb'].permute(1, 2, 0).numpy() * 255
        target = sample['qry_msk']
        ref_image = sample['sup_rgb'][0].permute(1, 2, 0).numpy() * 255
        ref_mask = sample['sup_msk'][0:1].permute(1, 2, 0).numpy()
        image = image.astype(np.uint8)
        ref_image = ref_image.astype(np.uint8)
        sample_cls = sample['cls']

        res, hres = 448, 448
        input_image = image
        image = Image.fromarray(image)
        size = image.size
        image = np.array(image.resize((res, hres))) / 255.

        img2 = Image.fromarray(ref_image).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.

        print(ref_mask.dtype)
        ref_mask[ref_mask == 255] = 0
        ref_mask *= 255
        tgt2 = Image.fromarray(ref_mask[:, :, 0].astype(np.uint8)).convert("RGB")
        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.

        tgt = tgt2

        tgt = np.concatenate((tgt2, tgt), axis=0)
        img = np.concatenate((img2, image), axis=0)

        img = img - imagenet_mean
        img = img / imagenet_std

        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        img = img[None, ...]
        tgt = tgt[None, ...]

        output = run_one_image(img, tgt, model, device)
        output = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2),
            size=[size[1], size[0]],
            mode='nearest',
        ).permute(0, 2, 3, 1)[0].numpy()

        output = (0.6 * output / 255 + 0.4)
        output = output.mean(2) > 0.5
        print(np.unique(output))
        # output = Image.fromarray((input_image * (0.6 * output / 255 + 0.4)).astype(np.uint8))
        metric_m1.update(output[None, ...], target[None, ...], [sample['cls']], verbose=False)
        miou_class_1, miou_avg_1 = metric_m1.get_scores(datasets.get_val_labels(opt, None))
        bar.set_description(f"mIoU 1: {miou_avg_1} class_miou: {miou_class_1}")

        plt.figure(figsize=(20, 20))
        plt.subplot(2, 2, 1)
        plt.imshow(output)
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.imshow(input_image)
        plt.axis('off')

        plt.subplot(2, 2, 3)
        ref_mask = tgt2
        plt.imshow(ref_mask)
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(ref_image)

        plt.axis('off')
        plt.savefig('result.png')
        plt.close()


if __name__ == '__main__':
    ex.run_commandline()
