_base_ = ["../../_base_/models/swin/swin_base.py", "../../_base_/default_runtime.py"]

# dataset settings
dataset_type = "VideoDataset"
data_root = "DATASET/papadamou"
data_root_val = "DATASET/papadamou"
ann_file_train = "DATASET/papadamou/papadamou_train_list_videos.txt"
ann_file_val = "DATASET/papadamou/papadamou_val_list_videos.txt"
ann_file_test = "DATASET/papadamou/papadamou_test_list_videos.txt"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False
)
train_pipeline = [
    dict(type="DecordInit"),
    dict(
        type="SampleFrames",
        clip_len=32,
        frame_interval=30,
        num_clips=1,
        # frame_uniform=True,
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0),
    # dict(type="Imgaug", transforms=[dict(type="RandAugment", n=4, m=7)]),
    dict(type="Normalize", **img_norm_cfg),
    # dict(type="RandomErasing", probability=0.25),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]
val_pipeline = [
    dict(type="DecordInit"),
    dict(
        type="SampleFrames",
        clip_len=32,
        frame_interval=30,
        num_clips=1,
        # frame_uniform=True,
        test_mode=True,
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
test_pipeline = [
    dict(type="DecordInit"),
    dict(
        type="SampleFrames",
        clip_len=32,
        frame_interval=30,
        num_clips=1,
        # frame_uniform=True,
        test_mode=True,
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="ThreeCrop", crop_size=224),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(
    interval=5, metrics=["top_k_accuracy", "mean_class_accuracy"], gpu_collect=True
)

# optimizer
# optimizer = dict(
#     type="AdamW",
#     lr=5e-4,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             "absolute_pos_embed": dict(decay_mult=0.0),
#             "relative_position_bias_table": dict(decay_mult=0.0),
#             "norm": dict(decay_mult=0.0),
#             "backbone": dict(lr_mult=0.1),
#         }
#     ),
# )
# optimizer
lw_lr_decay = 0.9
optimizer = dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.001,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone.norm': dict(lr_mult=lw_lr_decay, decay_mult=0.),
                                                 'backbone.layers.3.blocks.1': dict(lr_mult=lw_lr_decay),
                                                 'backbone.layers.3.blocks.0': dict(lr_mult=lw_lr_decay ** 2),
                                                 'backbone.layers.2.downsample': dict(lr_mult=lw_lr_decay ** 3),
                                                 'backbone.layers.2.blocks.17': dict(lr_mult=lw_lr_decay ** 4),
                                                 'backbone.layers.2.blocks.16': dict(lr_mult=lw_lr_decay ** 5),
                                                 'backbone.layers.2.blocks.15': dict(lr_mult=lw_lr_decay ** 6),
                                                 'backbone.layers.2.blocks.14': dict(lr_mult=lw_lr_decay ** 7),
                                                 'backbone.layers.2.blocks.13': dict(lr_mult=lw_lr_decay ** 8),
                                                 'backbone.layers.2.blocks.12': dict(lr_mult=lw_lr_decay ** 9),
                                                 'backbone.layers.2.blocks.11': dict(lr_mult=lw_lr_decay ** 10),
                                                 'backbone.layers.2.blocks.10': dict(lr_mult=lw_lr_decay ** 11),
                                                 'backbone.layers.2.blocks.9': dict(lr_mult=lw_lr_decay ** 12),
                                                 'backbone.layers.2.blocks.8': dict(lr_mult=lw_lr_decay ** 13),
                                                 'backbone.layers.2.blocks.7': dict(lr_mult=lw_lr_decay ** 14),
                                                 'backbone.layers.2.blocks.6': dict(lr_mult=lw_lr_decay ** 15),
                                                 'backbone.layers.2.blocks.5': dict(lr_mult=lw_lr_decay ** 16),
                                                 'backbone.layers.2.blocks.4': dict(lr_mult=lw_lr_decay ** 17),
                                                 'backbone.layers.2.blocks.3': dict(lr_mult=lw_lr_decay ** 18),
                                                 'backbone.layers.2.blocks.2': dict(lr_mult=lw_lr_decay ** 19),
                                                 'backbone.layers.2.blocks.1': dict(lr_mult=lw_lr_decay ** 20),
                                                 'backbone.layers.2.blocks.0': dict(lr_mult=lw_lr_decay ** 21),
                                                 'backbone.layers.1.downsample': dict(lr_mult=lw_lr_decay ** 22),
                                                 'backbone.layers.1.blocks.1': dict(lr_mult=lw_lr_decay ** 23),
                                                 'backbone.layers.1.blocks.0': dict(lr_mult=lw_lr_decay ** 24),
                                                 'backbone.layers.0.downsample': dict(lr_mult=lw_lr_decay ** 25),
                                                 'backbone.layers.0.blocks.1': dict(lr_mult=lw_lr_decay ** 26),
                                                 'backbone.layers.0.blocks.0': dict(lr_mult=lw_lr_decay ** 27),
                                                 'backbone.patch_embed': dict(lr_mult=lw_lr_decay ** 28),
                                                 }))

# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=0,
    warmup="linear",
    warmup_by_epoch=True,
    warmup_iters=2.5,
)
total_epochs = 60

# runtime settings
checkpoint_config = dict(interval=5, create_symlink=False)
auto_resume = True
work_dir = "OUTPUT/swin_base_bevt_finetune_videomisinfo_minute"
find_unused_parameters = False
custom_hooks = [dict(type="EMAHook")]

fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=8,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

model = dict(
    backbone=dict(
        patch_size=(2, 4, 4),
        drop_path_rate=0.3,
        pretrained2d=False,
        # pretrained_window_size=(8, 7, 7),
    ),
    cls_head=dict(num_classes=2),
    test_cfg=dict(max_testing_views=2),
    train_cfg=dict(blending=dict(type="LabelSmoothing", num_classes=2, smoothing=0.1)),
)
