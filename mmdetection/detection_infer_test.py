import torch
import torchvision
import torchvision.transforms as T
import colorsys
import random
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import Union, Optional, List, Tuple

transforms = []
transforms.append(T.ToTensor())
tfms = T.Compose(transforms)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# test_image = '/Users/derek/Downloads/PennFudanPed/PNGImages/FudanPed00011.png'
test_image = '/Users/derek/PycharmProjects/towhee/towhee/tests/dataset/kaggle_dataset_small/train' \
             '/00214f311d5d2247d5dfe4fe24b2303d.jpg'
img_pil = Image.open(test_image).convert('RGB')
img_tensor = torchvision.transforms.ToTensor()(img_pil)
images = [img_tensor.to(device)]

predictions = model(images)
pred = predictions[0]
print(pred)

scores = pred['scores']
mask = scores >= 0.9

boxes = pred['boxes'][mask]
labels = pred['labels'][mask]
scores = scores[mask]


def golden(n, h=random.random(), s=0.5, v=0.95,
           fn=None, scale=None, shuffle=False):
    if n <= 0:
        return []

    coef = (1 + 5 ** 0.5) / 2

    colors = []
    for _ in range(n):
        h += coef
        h = h - int(h)
        color = colorsys.hsv_to_rgb(h, s, v)
        if scale is not None:
            color = tuple(scale * v for v in color)
        if fn is not None:
            color = tuple(fn(v) for v in color)
        colors.append(color)

    if shuffle:
        random.shuffle(colors)
    return colors


def plot_image(
        image: Union[torch.Tensor, Image.Image, np.ndarray],
        boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        lb_names: Optional[List[str]] = None,
        lb_colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
        lb_infos: Optional[List[str]] = None,
        save_name: Optional[str] = None,
        show_name: Optional[str] = 'result',
) -> torch.Tensor:
    """
    Draws bounding boxes on given image.
    Args:
      image (Image): `Tensor`, `PIL Image` or `numpy.ndarray`.
      boxes (Optional[Tensor]): `FloatTensor[N, 4]`, the boxes in `[x1, y1, x2, y2]` format.
      labels (Optional[Tensor]): `Int64Tensor[N]`, the class label index for each box.
      lb_names (Optional[List[str]]): All class label names.
      lb_colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of all class label names.
      lb_infos (Optional[List[str]]): Infos for given labels.
      save_name (Optional[str]): Save image name.
      show_name (Optional[str]): Show window name.
    """
    if not isinstance(image, torch.Tensor):
        image = torchvision.transforms.ToTensor()(image)

    if boxes is not None:
        if image.dtype != torch.uint8:
            image = torchvision.transforms.ConvertImageDtype(torch.uint8)(image)
        draw_labels = None
        draw_colors = None
        if labels is not None:
            draw_labels = [lb_names[i] for i in labels] if lb_names is not None else None
            draw_colors = [lb_colors[i] for i in labels] if lb_colors is not None else None
        if draw_labels and lb_infos:
            draw_labels = [f'{l} {i}' for l, i in zip(draw_labels, lb_infos)]
        # torchvision >= 0.9.0/nightly
        #  https://github.com/pytorch/vision/blob/master/torchvision/utils.py
        res = torchvision.utils.draw_bounding_boxes(image, boxes,
                                                    labels=draw_labels, colors=draw_colors)
    else:
        res = image

    if save_name or show_name:
        res = res.permute(1, 2, 0).contiguous().numpy()
        if save_name:
            Image.fromarray(res).save(save_name)
        if show_name:
            plt.gcf().canvas.set_window_title(show_name)
            plt.imshow(res)
            plt.show()

    return res


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
lb_names = COCO_INSTANCE_CATEGORY_NAMES
lb_colors = golden(len(lb_names), fn=int, scale=0xff, shuffle=True)
lb_infos = [f'{s:.2f}' for s in scores]
plot_image(img_tensor, boxes, labels, lb_names, lb_colors, lb_infos,
           save_name='result.png')
