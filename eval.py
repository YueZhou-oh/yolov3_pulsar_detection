"""YoloV3 eval."""
import os
import argparse
import datetime
import time
import sys
from collections import defaultdict

import numpy as np

from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore as ms

from src.yolo import YOLOV3DarkNet53
from src.logger import get_logger
from src.yolo_dataset import create_yolo_dataset
from src.config import ConfigYOLOV3DarkNet53

class DetectionEngine:
    """Detection engine."""
    def __init__(self, args):
        self.results = {}
        self.file_path = ''
        self.save_prefix = args.outputs_dir
        self.annFile = args.annFile
        self.det_boxes = []
        self.args = args
    
    def _nms(self, dets, scores, thresh, score_thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        order = order[scores[order] > score_thresh]

        keep = []
        inds = 0
        while order.size > 0:
            i = order.item(0)
            # print(i)
            if areas[i] > 200 and areas[i] < 50000:
                keep.append(i)
#             else:
#                 # print(inds)
#                 order = order[inds + 1]
#                 # order = order[inds]
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def write_result(self):
        """Save result to file."""
        import json
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            print("saving to {}...".format(self.file_path))
            f = open(self.file_path, 'w')
            json.dump(self.det_boxes, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def detect(self, outputs, batch, image_shape, image_id, config):
        """Detect boxes."""
        outputs_num = len(outputs)
        for batch_id in range(batch):
            img_id = image_id[batch_id]
            img_id_name = "fake"
            for tmp in img_id:
                img_id_name = img_id_name + "_" + str(tmp)
            img_id_name =img_id_name + ".png"
            print("image name: {}".format(img_id_name))
            for out_id in range(outputs_num):
                out_item = outputs[out_id]
                out_item_single = out_item[batch_id, :]
                dimensions = out_item_single.shape[:-1]
                out_num = 1
                for d in dimensions:
                    out_num *= d
                [ori_h, ori_w] = image_shape
                x = out_item_single[..., 0] * ori_w
                y = out_item_single[..., 1] * ori_h
                w = out_item_single[..., 2] * ori_w
                h = out_item_single[..., 3] * ori_h

                conf = out_item_single[..., 4:5]
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                w = w.reshape(-1, 1)
                h = h.reshape(-1, 1)
                conf = conf.reshape(-1)

                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                x_bottom_right = x + w / 2.
                y_bottom_right = y + h / 2.

                inds_1 = np.where(np.logical_and(x_top_left <= ori_w , x_top_left >= 0))[0]
                inds_2 = np.where(np.logical_and(y_top_left <= ori_h , y_top_left >= 0))[0]
                inds = np.intersect1d(inds_1, inds_2)
                inds_3 = np.where(np.logical_and(x_bottom_right <= ori_w , x_bottom_right >= 0))[0]
                inds = np.intersect1d(inds, inds_3)
                inds_4 = np.where(np.logical_and(y_bottom_right <= ori_h , y_bottom_right >= 0))[0]
                inds = np.intersect1d(inds, inds_4)

                bbox = np.concatenate((x_top_left[inds], y_top_left[inds], x_bottom_right[inds], y_bottom_right[inds]), axis=1)
                conf = conf[inds]
                if out_id == 0:
                    bbox_cor = bbox
                    bbox_score = conf
                else:
                    bbox_cor = np.append(bbox_cor, bbox, axis = 0)
                    bbox_score = np.append(bbox_score, conf, axis = 0)

            keep = self._nms(bbox_cor, bbox_score, config.iou_thr, config.score_thr)
            keep = keep[:config.max_box]
            keep_bbox = bbox_cor[keep]
            keep_bbox_score = bbox_score[keep]
            self.args.logger.info("detected objects: {}".format(keep_bbox.shape[0]))
            self.args.logger.info("bbox: {}".format(keep_bbox))
            self.args.logger.info("score: {}".format(keep_bbox_score))
            self.args.logger.info("================================================")

            # print("detected objects: ", keep_bbox.shape[0])
            # print("bbox: ", keep_bbox)
            # print("score: ", keep_bbox_score)
            # print("================================================")
            tmp = [{
                "image_name": img_id_name,
                'bbox': keep_bbox.tolist(),
                'score': keep_bbox_score.tolist()
            }]
            self.det_boxes.extend(tmp)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser('mindspore coco testing')

    # device related
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')

    # dataset related
    parser.add_argument('--data_dir', type=str, default='/home/ma-user/work/pulsar/pulsar-data/fake_images/fake_pulsar_test_2', help='train data dir')
#     parser.add_argument('--data_dir', type=str, default='/home/ma-user/work/pulsar/pulsar-data/fake_images/tmp', help='train data dir')
    parser.add_argument('--per_batch_size', default=1, type=int, help='batch size for per gpu')

    # network related
#     parser.add_argument('--pretrained', default='/home/ma-user/work/yolov3_pulsar/outputs/2021-05-26_time_06_43_09/ckpt_0/0-60_15000.ckpt', type=str, help='model_path, local pretrained model to load')  # anchor : [h, w]
#     parser.add_argument('--pretrained', default='/home/ma-user/work/yolov3_pulsar/outputs/2021-06-01_time_03_50_18/ckpt_0/0-60_15000.ckpt', type=str, help='model_path, local pretrained model to load')  # anchor : [w, h]
    parser.add_argument('--pretrained', default='/home/ma-user/work/yolov3_pulsar/outputs/2021-06-02_time_06_39_19/ckpt_0/0-60_15000.ckpt', type=str, help='model_path, local pretrained model to load')  # anchor : [h, w], network [h, w]
#     parser.add_argument('--pretrained', default='/home/ma-user/work/yolov3_pulsar/outputs/2021-05-31_time_11_01_08/ckpt_0/0-60_15000.ckpt', type=str, help='model_path, local pretrained model to load')


    # logging related
    parser.add_argument('--log_path', type=str, default='outputs/', help='checkpoint save location')

    # detect_related
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='threshold for NMS')
    parser.add_argument('--annFile', type=str, default='/home/ma-user/work/pulsar/pulsar-data/annotations/fake_pulsar_test_2_ann.json', help='path to annotation')


    args, _ = parser.parse_known_args()

    return args


def test():
    """The function of eval."""
    start_time = time.time()
    args = parse_args()

    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=True, device_id=devid)

    # logger
    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    args.logger = get_logger(args.outputs_dir, rank_id)

    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    args.logger.info('Creating Network....')
    network = YOLOV3DarkNet53(is_training=False)

    args.logger.info(args.pretrained)
    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained))
    else:
        args.logger.info('{} not exists or not a pre-trained file'.format(args.pretrained))
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(args.pretrained))
        exit(1)

    config = ConfigYOLOV3DarkNet53()

    ds, data_size = create_yolo_dataset(args.data_dir, args.annFile, batch_size=args.per_batch_size,
                                        max_epoch=1, device_num=1, rank=rank_id,
                                        config=config, is_training=False)

    args.logger.info('testing shape : {}'.format(config.test_img_shape))
    args.logger.info('total {} images to eval'.format(data_size))

    network.set_train(False)

    # init detection engine
    detection = DetectionEngine(args)

    input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
    args.logger.info('Start inference....')
    for i, data in enumerate(ds.create_dict_iterator()):
        image = data["img"]
        image_id = data["img_id"]
        anno = data["ann"]

        prediction = network(image, input_shape)
        output_big, output_me, output_small = prediction
        image_id = image_id.asnumpy()
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        image_shape = config.test_img_shape

        detection.detect([output_small, output_me, output_big], args.per_batch_size, image_shape, image_id, config)
        if i % 1000 == 0:
            args.logger.info('Processing... {:.2f}% '.format(i * args.per_batch_size / data_size * 100))

    args.logger.info('Calculating mAP...')
    result_file_path = detection.write_result()
    args.logger.info('result file path: {}'.format(result_file_path))

    cost_time = time.time() - start_time
    args.logger.info('testing cost time {:.2f}h'.format(cost_time / 3600.))


if __name__ == "__main__":
    test()
