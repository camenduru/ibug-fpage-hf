#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import sys
import tarfile

import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import torch

sys.path.insert(0, 'face_detection')
sys.path.insert(0, 'face_parsing')
sys.path.insert(0, 'fpage')
sys.path.insert(0, 'roi_tanh_warping')

from ibug.age_estimation import AgeEstimator
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing.utils import label_colormap

REPO_URL = 'https://github.com/ibug-group/fpage'
TITLE = 'ibug-group/fpage'
DESCRIPTION = f'This is a demo for {REPO_URL}.'
ARTICLE = None

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def load_sample_images() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        image_dir.mkdir()
        dataset_repo = 'hysts/input-images'
        filenames = ['003.tar']
        for name in filenames:
            path = huggingface_hub.hf_hub_download(dataset_repo,
                                                   name,
                                                   repo_type='dataset',
                                                   use_auth_token=TOKEN)
            with tarfile.open(path) as f:
                f.extractall(image_dir.as_posix())
    return sorted(image_dir.rglob('*.jpg'))


def load_detector(device: torch.device) -> RetinaFacePredictor:
    model = RetinaFacePredictor(
        threshold=0.8,
        device=device,
        model=RetinaFacePredictor.get_model('mobilenet0.25'))
    return model


def load_model(device: torch.device) -> AgeEstimator:
    ckpt_path = huggingface_hub.hf_hub_download(
        'hysts/ibug',
        'fpage/models/fpage-resnet50-fcn-14-97.torch',
        use_auth_token=TOKEN)
    model = AgeEstimator(
        device=device,
        ckpt=ckpt_path,
        encoder='resnet50',
        decoder='fcn',
        age_classes=97,
        face_classes=14,
    )
    return model


def predict(image: np.ndarray, max_num_faces: int,
            detector: RetinaFacePredictor, model: AgeEstimator) -> np.ndarray:
    colormap = label_colormap(14)

    # RGB -> BGR
    image = image[:, :, ::-1]

    faces = detector(image, rgb=False)
    if len(faces) == 0:
        raise RuntimeError('No face was found.')
    faces = sorted(list(faces), key=lambda x: -x[4])[:max_num_faces][::-1]
    ages, masks = model.predict_img(image, faces, rgb=False)

    mask_image = np.zeros_like(image)
    for mask in masks:
        temp = colormap[mask]
        mask_image[temp > 0] = temp[temp > 0]

    res = image.astype(float) * 0.5 + mask_image[:, :, ::-1] * 0.5
    res = np.clip(np.round(res), 0, 255).astype(np.uint8)

    for face, age in zip(faces, ages):
        bbox = np.round(face[:4]).astype(int)
        cv2.rectangle(
            res,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color=(0, 255, 0),
            thickness=2,
        )

        text_content = f'Age: ({age: .1f})'
        cv2.putText(
            res,
            text_content,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 255, 255),
            lineType=cv2.LINE_AA,
        )

    return res[:, :, ::-1]


def main():
    args = parse_args()
    device = torch.device(args.device)

    detector = load_detector(device)
    model = load_model(device)

    func = functools.partial(predict, detector=detector, model=model)
    func = functools.update_wrapper(func, predict)

    image_paths = load_sample_images()
    examples = [[path.as_posix(), 5] for path in image_paths]

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='numpy', label='Input'),
            gr.inputs.Slider(
                1, 20, step=1, default=5, label='Max Number of Faces'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
