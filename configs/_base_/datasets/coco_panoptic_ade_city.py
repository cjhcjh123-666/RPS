# dataset settings
from mmengine import read_base
from mmengine.dataset import DefaultSampler, RepeatDataset

from seg.datasets.concat_dataset import ConcatOVDataset
from seg.datasets.samplers.batch_sampler import VideoSegAspectRatioBatchSampler

with read_base():
    from .coco_panoptic_lsj import train_dataloader as _coco_train_loader
    from .ade_panoptic_ov import train_dataloader as _ade_train_loader
    from .cityscapes_panoptic import train_dataloader as _city_train_loader
from mmcv.transforms import LoadImageFromFile, RandomResize
from mmengine.dataset import DefaultSampler

from mmdet.datasets.transforms import LoadPanopticAnnotations, RandomFlip, RandomCrop, PackDetInputs, Resize

from seg.datasets.coco_ov import CocoPanopticOVDataset
from seg.datasets.pipeliens.loading import FilterAnnotationsHB
from seg.datasets.pipeliens.formatting import GeneratePoint
from seg.datasets.pipeliens.loading import LoadPanopticAnnotationsAll

image_size = (1024, 1024)
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=VideoSegAspectRatioBatchSampler),
    dataset=dict(
        type=ConcatOVDataset,
        data_tag=('coco', 'ade', 'city'),
        datasets=[
            dict(
                type=RepeatDataset,
                dataset=_coco_train_loader.dataset,
                times=1,
            ),
            dict(
                type=RepeatDataset,
                dataset=_ade_train_loader.dataset,
                times=25,
            ),
            dict(
                type=RepeatDataset,
                dataset=_city_train_loader.dataset,
                times=1,
            )
        ]
    ),
)
