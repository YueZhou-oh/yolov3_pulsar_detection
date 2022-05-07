"""Config parameters for Darknet based yolov3_darknet53 models."""


class ConfigYOLOV3DarkNet53:
    """
    Config parameters for the yolov3_darknet53.

    Examples:
        ConfigYOLOV3DarkNet53()
    """
    # train_param
    # data augmentation related
    hue = 0.1
    saturation = 1.5
    value = 1.5
    jitter = 0.3
    brightness = 0.5
    contrast = 0.5
    
    iou_thr = 0.1
    score_thr = 0.5
    resize_rate = 1
    # [height, width]
    multi_scale = [[224, 3840]
                   ]
#     multi_scale = [[224, 3840],
#                    [256, 4160],
#                    [288, 4480],
#                    [320, 4800]
#                    ]

    scales = len(multi_scale)
    num_classes = 2
    max_box = 20

    backbone_input_shape = [32, 64, 128, 256, 512]
    backbone_shape = [64, 128, 256, 512, 1024]
    backbone_layers = [1, 2, 8, 8, 4]

    # confidence under ignore_threshold means no object when training
    ignore_threshold = 0.7

    # h->w, greater than or equal to scales*3, smaller than scales*4
    anchor_scales = [(50, 6),
                     (100, 10),
                     (100, 30),
                     (80, 80),
                     (150, 50),
                     (175, 100),
                     (150, 120),
                     (150, 150),
                     (200, 200)]

    out_channel = 3 * (num_classes + 5)   # 255

    # [height, width]
    test_img_shape = [224, 3840]

    data_pre = {
        'type': 'train',
        'file_type': 'img',
        'num_works': 4,
        'pin_memory': True,
        'num_samps' : 32,
        'fch1': 1499.93896484375,
        'fend': 1000,
        'scale_tsamp': 0.006291456,
    }
