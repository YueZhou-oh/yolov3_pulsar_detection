"""Preprocess dataset."""
import random
import threading
import copy
import mindspore as ms
from mindspore import Tensor
import numpy as np
from PIL import Image, ImageEnhance
import cv2

class Transformer():
    def __init__(self,
                 scale_tsamp=0.006291456,
                 scale_fsamp=1.9531,
                 fch1=1499.93896484375,
                 fend=1000,
                 padding=1):
        self.scale_tsamp = scale_tsamp
        self.scale_fsamp = scale_fsamp
        self.fch1 = fch1
        self.fend = fend
        self.padding = padding
        self.delay_constant = 4148.808

    def img_detrans(self, img):
        # img = img * self.norm_param[1] + self.norm_param[0]
        img = img*255
        img = img.astype(np.uint8)
        return img

    def ann_trans(self, anns, num_samps=None):
        '''

        :param anns: (numpy.array) N * 5 [t0, DM, f0, f1, isfake]
        :return: (tensor)
        '''
        if len(anns) == 0:
            return anns

        perm = (1, 0)

        f0 = anns[:, 2]
        f1 = anns[:, 3]

        dt0 = self.delay_constant * (1/f0**2 - 1/self.fch1**2) * anns[:,1]
        x0 = np.round((anns[:, 0] + dt0) / self.scale_tsamp)

        y1 = np.round((self.fch1 - f1) / self.scale_fsamp)
        y0 = np.round((self.fch1 - f0) / self.scale_fsamp)

        dt = self.delay_constant * (1/f1**2 - 1/self.fch1**2) * anns[:,1]
        x1 = np.round((anns[:,0] + dt) / self.scale_tsamp)

        f_center = (f0 + f1) /2
        y_center = np.round((self.fch1-f_center) / self.scale_fsamp)
        x_center = np.round((anns[:,0] + self.delay_constant * (1/f_center**2 - 1/self.fch1**2)) / self.scale_tsamp)
        anns = np.array([x0, y0, x1, y1, np.ones_like(x_center)], dtype=np.float32)
        anns = anns.transpose(perm)
        return anns


def _rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.

    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def statistic_normalize_img(img, statistic_norm):
    """Statistic normalize images."""
    # img: RGB
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img/255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if statistic_norm:
        img = (img - mean) / std
    return img

def normalize_img(img):
    """normalize images."""
    # img: RGB
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img/255.
    return img

def get_interp_method(interp, sizes=()):
    """
    Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic or Bilinear.

    Args:
        interp (int): Interpolation method for all resizing operations.

            - 0: Nearest Neighbors Interpolation.
            - 1: Bilinear interpolation.
            - 2: Bicubic interpolation over 4x4 pixel neighborhood.
            - 3: Nearest Neighbors. Originally it should be Area-based, as we cannot find Area-based,
              so we use NN instead. Area-based (resampling using pixel area relation).
              It may be a preferred method for image decimation, as it gives moire-free results.
              But when the image is zoomed, it is similar to the Nearest Neighbors method. (used by default).
            - 4: Lanczos interpolation over 8x8 pixel neighborhood.
            - 9: Cubic for enlarge, area for shrink, bilinear for others.
            - 10: Random select from interpolation method mentioned above.

        sizes (tuple): Format should like (old_height, old_width, new_height, new_width),
            if None provided, auto(9) will return Area(2) anyway. Default: ()

    Returns:
        int, interp method from 0 to 4.
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            if nh < oh and nw < ow:
                return 0
            return 1
        return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp


def pil_image_reshape(interp):
    """Reshape pil image."""
    reshape_type = {
        0: Image.NEAREST,
        1: Image.BILINEAR,
        2: Image.BICUBIC,
        3: Image.NEAREST,
        4: Image.LANCZOS,
    }
    return reshape_type[interp]


def _preprocess_true_boxes(true_boxes, anchors, in_shape, num_classes, max_boxes, label_smooth,
                           label_smooth_factor=0.1):
    """
    Introduction
    ------------
        对训练数据的ground truth box进行预处理
    Parameters
    ----------
        true_boxes: ground truth box 形状为[boxes, 5], x_min, y_min, x_max, y_max, class_id
    """
    anchors = np.array(anchors)
    num_layers = anchors.shape[0] // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(in_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.  # center
    # trans to box center point
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # input_shape is [h, w], normalization
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
    # true_boxes = [xywh]
    #print(true_boxes)
    grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
    # grid_shape [h, w]
    y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]),
                        5 + num_classes), dtype='float32') for l in range(num_layers)]
    # y_true [gridy, gridx]
    # 这里扩充维度是为了后面应用广播计算每个图中所有box的anchor互相之间的iou
    anchors = np.expand_dims(anchors, 0)
    anchors_max = anchors / 2.
    anchors_min = -anchors_max
    # 因为之前对box做了padding, 因此需要去除全0行
    valid_mask = boxes_wh[..., 0] > 0

    wh = boxes_wh[valid_mask]
    if wh.size > 0:
        # 为了应用广播扩充维度
        wh = np.expand_dims(wh, -2)
        # wh 的shape为[box_num, 1, 2]
        # move to original point to compare, and choose the best layer-anchor to set
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        intersect_min = np.maximum(boxes_min, anchors_min)
        intersect_max = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # 找出和ground truth box的iou最大的anchor box,
        # 然后将对应不同比例的负责该ground turth box 的位置置为ground truth box坐标
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')  # grid_y
                    j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')  # grid_x

                    k = anchor_mask[l].index(n)
                    c = true_boxes[t, 4].astype('int32')
                    y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                    y_true[l][j, i, k, 4] = 1.

                    # lable-smooth
                    if label_smooth:
                        sigma = label_smooth_factor / (num_classes - 1)
                        y_true[l][j, i, k, 5:] = sigma
                        y_true[l][j, i, k, 5 + c] = 1 - label_smooth_factor
                    else:
                        y_true[l][j, i, k, 5 + c] = 1.

    # pad_gt_boxes for avoiding dynamic shape
    pad_gt_box0 = np.zeros(shape=[max_boxes, 4], dtype=np.float32)
    pad_gt_box1 = np.zeros(shape=[max_boxes, 4], dtype=np.float32)
    pad_gt_box2 = np.zeros(shape=[max_boxes, 4], dtype=np.float32)

    mask0 = np.reshape(y_true[0][..., 4:5], [-1])
    gt_box0 = np.reshape(y_true[0][..., 0:4], [-1, 4])
    # gt_box [boxes, [x,y,w,h]]
    gt_box0 = gt_box0[mask0 == 1]
    # gt_box0: get all boxes which have object
    if gt_box0.shape[0] < max_boxes:
        pad_gt_box0[:gt_box0.shape[0]] = gt_box0
    else:
        pad_gt_box0 = gt_box0[:max_boxes]
    # gt_box0.shape[0]: total number of boxes in gt_box0
    # top N of pad_gt_box0 is real box, and after are pad by zero

    mask1 = np.reshape(y_true[1][..., 4:5], [-1])
    gt_box1 = np.reshape(y_true[1][..., 0:4], [-1, 4])
    gt_box1 = gt_box1[mask1 == 1]
    if gt_box1.shape[0] < max_boxes:
        pad_gt_box1[:gt_box1.shape[0]] = gt_box1
    else:
        pad_gt_box1 = gt_box1[:max_boxes]

    mask2 = np.reshape(y_true[2][..., 4:5], [-1])
    gt_box2 = np.reshape(y_true[2][..., 0:4], [-1, 4])

    gt_box2 = gt_box2[mask2 == 1]
    if gt_box2.shape[0] < max_boxes:
        pad_gt_box2[:gt_box2.shape[0]] = gt_box2
    else:
        pad_gt_box2 = gt_box2[:max_boxes]
    return y_true[0], y_true[1], y_true[2], pad_gt_box0, pad_gt_box1, pad_gt_box2

# not called during training process
def _reshape_data(image, box, input_size):
    """image preprocessing."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    ori_w, ori_h = image.size
    input_h, input_w = input_size

    w_ratio = input_w * 1.0 / ori_w
    h_ratio = input_h * 1.0 / ori_h

    interp = get_interp_method(interp=9, sizes=(ori_h, ori_w, input_h, input_w))
    image = image.resize((input_w, input_h), pil_image_reshape(interp))

    image = normalize_img(image)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        image = np.concatenate([image, image, image], axis=-1)

    image = np.transpose(image, [2, 0, 1])
    image = image.astype(np.float32)
    np.random.shuffle(box)

    t_box = copy.deepcopy(box)
    t_box[:, [0, 2]] = t_box[:, [0, 2]] * w_ratio
    t_box[:, [1, 3]] = t_box[:, [1, 3]] * h_ratio

    return image, t_box



def color_distortion(img, hue, sat, val, device_num):
    """Color distortion."""
    hue = _rand(-hue, hue)
    sat = _rand(1, sat) if _rand() < .5 else 1 / _rand(1, sat)
    val = _rand(1, val) if _rand() < .5 else 1 / _rand(1, val)
    if device_num != 1:
        cv2.setNumThreads(1)
    x = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    x = x / 255.
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    x = x * 255.
    x = x.astype(np.uint8)
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB_FULL)
    return image_data


def filp_pil_image(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def convert_gray_to_color(img):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
        img = np.concatenate([img, img, img], axis=-1)
    return img

def _data_aug(image, box, brightness, contrast, image_input_size, max_boxes):

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image_w, image_h = image.size
    input_h, input_w = image_input_size

    np.random.shuffle(box)
    if len(box) > max_boxes:
        box = box[:max_boxes]
    box = np.array(box, dtype=np.float16)
    # act_obj = box.shape[0]

    w_ratio = input_w * 1.0 / image_w
    h_ratio = input_h * 1.0 / image_h

    box[:, [0, 2]] = box[:, [0, 2]] * w_ratio
    box[:, [1, 3]] = box[:, [1, 3]] * h_ratio

    keep_w = np.logical_and(box[:, 0] < input_w, box[:, 2] <= input_w)
    keep_h = np.logical_and(box[:, 1] < input_h, box[:, 3] <= input_h)
    keep = np.tile(np.logical_and(keep_h, keep_w), (1, 5)).reshape(5, -1).transpose()
    box = box * keep

    act_obj = box.shape[0]
    box_data = np.zeros((max_boxes, 5))
    box_data[:act_obj] = box

    image = ImageEnhance.Brightness(image).enhance(np.random.uniform(1 - brightness, 1 + brightness))
    image = ImageEnhance.Contrast(image).enhance(np.random.uniform(1 - contrast, 1 + contrast))

    interp = get_interp_method(interp=9, sizes=(image_h, image_w, input_h, input_w))
    image = image.resize((input_w, input_h), pil_image_reshape(interp))

    image = normalize_img(image)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        image = np.concatenate([image, image, image], axis=-1)

    image = image.astype(np.float32)

    return image, box_data

def preprocess_fn(image, box, config, input_size, device_num):
    """Preprocess data function."""
    max_boxes = config.max_box
    brightness = config.brightness
    contrast = config.contrast
    image, anno = _data_aug(image, box, brightness=brightness, contrast=contrast,
                            image_input_size=input_size, max_boxes=max_boxes)
    return image, anno

def _padding_zero(ann, max_box):
    ann_tmp = np.zeros(shape=[max_box, 5], dtype=np.float32)
    real_box = ann.shape[0]
    ann_tmp[:real_box] = ann
    return ann_tmp

def reshape_fn(image, ann, config, img_id):
    input_size = config.test_img_shape
    image, ann = _reshape_data(image, ann, input_size)
    # ann = _padding_zero(ann, config.max_box)

    return image, ann, img_id

class MultiScaleTrans:
    """Multi scale transform."""
    def __init__(self, config, device_num):
        self.config = config
        # self.args = args
        self.seed = 0
        self.size_list = []
        self.resize_rate = config.resize_rate
        self.dataset_size = config.dataset_size
        self.size_dict = {}
        # self.scales = config.scales
        self.seed_num = int(1e6)
        self.seed_list = self.generate_seed_list(seed_num=self.seed_num)
        self.resize_count_num = int(np.ceil(self.dataset_size / self.resize_rate))
        self.device_num = device_num
        self.anchor_scales = config.anchor_scales
        self.num_classes = config.num_classes
        self.max_box = config.max_box
        self.label_smooth = config.label_smooth
        self.label_smooth_factor = config.label_smooth_factor

    def generate_seed_list(self, init_seed=1234, seed_num=int(1e6), seed_range=(1, 1000)):
        seed_list = []
        random.seed(init_seed)
        for _ in range(seed_num):
            seed = random.randint(seed_range[0], seed_range[1])
            seed_list.append(seed)
        return seed_list

    def __call__(self, imgs, annos, x1, x2, x3, x4, x5, x6, batchInfo):
        epoch_num = batchInfo.get_epoch_num()
        size_idx = int(batchInfo.get_batch_num() / self.resize_rate)
        seed_key = self.seed_list[(epoch_num * self.resize_count_num + size_idx) % self.seed_num]
        # print("seed key:{}".format(seed_key))
        ret_imgs = []
        ret_annos = []

        bbox1 = []
        bbox2 = []
        bbox3 = []
        gt1 = []
        gt2 = []
        gt3 = []

        if self.size_dict.get(seed_key, None) is None:
            random.seed(seed_key)
            new_size = random.choice(self.config.multi_scale)
            self.size_dict[seed_key] = new_size
        seed = seed_key

        input_size = self.size_dict[seed]
        for img, anno in zip(imgs, annos):
            img, anno = preprocess_fn(img, anno, self.config, input_size, self.device_num)
            ret_imgs.append(img.transpose(2, 0, 1).copy())
            # img, anno = reshape_fn(img, anno, input_size, self.config, is_training=True)
            # ret_imgs.append(img.copy())
            bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
                _preprocess_true_boxes(true_boxes=anno, anchors=self.anchor_scales, in_shape=img.shape[0:2],
                                       num_classes=self.num_classes, max_boxes=self.max_box,
                                       label_smooth=self.label_smooth, label_smooth_factor=self.label_smooth_factor)
            bbox1.append(bbox_true_1)
            bbox2.append(bbox_true_2)
            bbox3.append(bbox_true_3)
            gt1.append(gt_box1)
            gt2.append(gt_box2)
            gt3.append(gt_box3)
            ret_annos.append(0)
        return np.array(ret_imgs), np.array(ret_annos), np.array(bbox1), np.array(bbox2), np.array(bbox3), \
               np.array(gt1), np.array(gt2), np.array(gt3)


def thread_batch_preprocess_true_box(annos, config, input_shape, result_index, batch_bbox_true_1, batch_bbox_true_2,
                                     batch_bbox_true_3, batch_gt_box1, batch_gt_box2, batch_gt_box3):
    """Preprocess true box for multi-thread."""
    i = 0
    for anno in annos:
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
            _preprocess_true_boxes(true_boxes=anno, anchors=config.anchor_scales, in_shape=input_shape,
                                   num_classes=config.num_classes, max_boxes=config.max_box,
                                   label_smooth=config.label_smooth, label_smooth_factor=config.label_smooth_factor)
        batch_bbox_true_1[result_index + i] = bbox_true_1
        batch_bbox_true_2[result_index + i] = bbox_true_2
        batch_bbox_true_3[result_index + i] = bbox_true_3
        batch_gt_box1[result_index + i] = gt_box1
        batch_gt_box2[result_index + i] = gt_box2
        batch_gt_box3[result_index + i] = gt_box3
        i = i + 1


def batch_preprocess_true_box(annos, config, input_shape):
    """Preprocess true box with multi-thread."""
    batch_bbox_true_1 = []
    batch_bbox_true_2 = []
    batch_bbox_true_3 = []
    batch_gt_box1 = []
    batch_gt_box2 = []
    batch_gt_box3 = []
    threads = []

    step = 4
    for index in range(0, len(annos), step):
        for _ in range(step):
            batch_bbox_true_1.append(None)
            batch_bbox_true_2.append(None)
            batch_bbox_true_3.append(None)
            batch_gt_box1.append(None)
            batch_gt_box2.append(None)
            batch_gt_box3.append(None)
        step_anno = annos[index: index + step]
        t = threading.Thread(target=thread_batch_preprocess_true_box,
                             args=(step_anno, config, input_shape, index, batch_bbox_true_1, batch_bbox_true_2,
                                   batch_bbox_true_3, batch_gt_box1, batch_gt_box2, batch_gt_box3))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return np.array(batch_bbox_true_1), np.array(batch_bbox_true_2), np.array(batch_bbox_true_3), \
           np.array(batch_gt_box1), np.array(batch_gt_box2), np.array(batch_gt_box3)


def batch_preprocess_true_box_single(annos, config, input_shape):
    """Preprocess true boxes."""
    batch_bbox_true_1 = []
    batch_bbox_true_2 = []
    batch_bbox_true_3 = []
    batch_gt_box1 = []
    batch_gt_box2 = []
    batch_gt_box3 = []
    for anno in annos:
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
            _preprocess_true_boxes(true_boxes=anno, anchors=config.anchor_scales, in_shape=input_shape,
                                   num_classes=config.num_classes, max_boxes=config.max_box,
                                   label_smooth=config.label_smooth, label_smooth_factor=config.label_smooth_factor)
        batch_bbox_true_1.append(bbox_true_1)
        batch_bbox_true_2.append(bbox_true_2)
        batch_bbox_true_3.append(bbox_true_3)
        batch_gt_box1.append(gt_box1)
        batch_gt_box2.append(gt_box2)
        batch_gt_box3.append(gt_box3)

    return np.array(batch_bbox_true_1), np.array(batch_bbox_true_2), np.array(batch_bbox_true_3), \
           np.array(batch_gt_box1), np.array(batch_gt_box2), np.array(batch_gt_box3)
