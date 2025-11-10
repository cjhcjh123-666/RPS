# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCOPanoptic
from mmdet.registry import METRICS
from mmdet.evaluation.metrics.coco_panoptic_metric import CocoPanopticMetric
from mmdet.evaluation.functional import (INSTANCE_OFFSET, pq_compute_multi_core,
                          pq_compute_single_core)

try:
    import panopticapi
    from panopticapi.evaluation import VOID, PQStat
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    panopticapi = None
    id2rgb = None
    rgb2id = None
    VOID = None
    PQStat = None


@METRICS.register_module()
class ADEPanopticMetric(CocoPanopticMetric):
    """ADE20K panoptic segmentation evaluation metric.
    
    This metric fixes the KeyError issue caused by class name mismatch
    between dataset and COCO API by using all category IDs instead of
    filtering by class names.
    
    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        seg_prefix (str, optional): Path to the directory which contains the
            coco panoptic segmentation mask. It should be specified when
            evaluate. Defaults to None.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created.
            It should be specified when format_only is True. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        nproc (int): Number of processes for panoptic quality computing.
            Defaults to 32. When ``nproc`` exceeds the number of cpu cores,
            the number of cpu cores is used.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'ade_panoptic'

    def _compute_batch_pq_stats(self, data_samples: Sequence[dict]):
        """Process gts and predictions when ``outfile_prefix`` is not set, gts
        are from dataset or a json file which is defined by ``ann_file``.

        Intermediate results, ``pq_stats``, are computed here and put into
        ``self.results``.
        
        This method fixes the KeyError issue by using all category IDs
        instead of filtering by class names.
        """
        if self._coco_api is None:
            categories = dict()
            for id, name in enumerate(self.dataset_meta['classes']):
                isthing = 1 if name in self.dataset_meta['thing_classes']\
                    else 0
                categories[id] = {'id': id, 'name': name, 'isthing': isthing}
            label2cat = None
        else:
            categories = self.categories
            # 修复：不使用类别名称过滤，直接使用所有类别ID
            # 这样可以避免类别名称不匹配导致的KeyError
            cat_ids = self._coco_api.get_cat_ids()
            label2cat = {i: cat_id for i, cat_id in enumerate(cat_ids)}

        for data_sample in data_samples:
            # parse pred
            img_id = data_sample['img_id']
            segm_file = osp.basename(data_sample['img_path']).replace(
                '.jpg', '.png')
            result = self._parse_predictions(
                pred=data_sample,
                img_id=img_id,
                segm_file=segm_file,
                label2cat=label2cat)

            # parse gt
            gt = dict()
            gt['image_id'] = img_id
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['file_name'] = segm_file

            if self._coco_api is None:
                # get segments_info from data_sample
                seg_map_path = osp.join(self.seg_prefix, segm_file)
                pan_png = mmcv.imread(seg_map_path).squeeze()
                pan_png = pan_png[:, :, ::-1]
                pan_png = rgb2id(pan_png)
                segments_info = []

                for segment_info in data_sample['segments_info']:
                    id = segment_info['id']
                    label = segment_info['category']
                    mask = pan_png == id
                    isthing = categories[label]['isthing']
                    if isthing:
                        iscrowd = 1 if not segment_info['is_thing'] else 0
                    else:
                        iscrowd = 0

                    new_segment_info = {
                        'id': id,
                        'category_id': label,
                        'isthing': isthing,
                        'iscrowd': iscrowd,
                        'area': mask.sum()
                    }
                    segments_info.append(new_segment_info)
            else:
                # get segments_info from annotation file
                segments_info = self._coco_api.imgToAnns[img_id]

            gt['segments_info'] = segments_info

            pq_stats = pq_compute_single_core(
                proc_id=0,
                annotation_set=[(gt, result)],
                gt_folder=self.seg_prefix,
                pred_folder=self.seg_out_dir,
                categories=categories,
                backend_args=self.backend_args)

            self.results.append(pq_stats)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch. There
                are two cases:

                - When ``outfile_prefix`` is not provided, the elements in
                  results are pq_stats which can be summed directly to get PQ.
                - When ``outfile_prefix`` is provided, the elements in
                  results are tuples like (gt, pred).

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.tmp_dir is None:
            # do evaluation after collect all the results

            # split gt and prediction list
            gts, preds = zip(*results)

            if self._coco_api is None:
                # use converted gt json file to initialize coco api
                logger.info('Converting ground truth to coco format...')
                coco_json_path, gt_folder = self.gt_to_coco_json(
                    gt_dicts=gts, outfile_prefix=self.outfile_prefix)
                self._coco_api = COCOPanoptic(coco_json_path)
            else:
                gt_folder = self.seg_prefix

            # 修复：不使用类别名称过滤，直接使用所有类别ID
            self.cat_ids = self._coco_api.get_cat_ids()
            self.cat2label = {
                cat_id: i
                for i, cat_id in enumerate(self.cat_ids)
            }
            self.img_ids = self._coco_api.get_img_ids()
            self.categories = self._coco_api.cats

            # convert predictions to coco format and dump to json file
            json_filename, pred_folder = self.result2json(
                results=preds, outfile_prefix=self.outfile_prefix)

            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(self.outfile_prefix)}')
                return dict()

            imgs = self._coco_api.imgs
            gt_json = self._coco_api.img_ann_map
            gt_json = [{
                'image_id': k,
                'segments_info': v,
                'file_name': imgs[k]['segm_file']
            } for k, v in gt_json.items()]
            pred_json = load(json_filename)
            pred_json = dict(
                (el['image_id'], el) for el in pred_json['annotations'])

            # match the gt_anns and pred_anns in the same image
            matched_annotations_list = []
            for gt_ann in gt_json:
                img_id = gt_ann['image_id']
                if img_id not in pred_json.keys():
                    raise Exception('no prediction for the image'
                                    ' with id: {}'.format(img_id))
                matched_annotations_list.append((gt_ann, pred_json[img_id]))

            pq_stat = pq_compute_multi_core(
                matched_annotations_list,
                gt_folder,
                pred_folder,
                self.categories,
                backend_args=self.backend_args,
                nproc=self.nproc)

        else:
            # aggregate the results generated in process
            if self._coco_api is None:
                categories = dict()
                for id, name in enumerate(self.dataset_meta['classes']):
                    isthing = 1 if name in self.dataset_meta[
                        'thing_classes'] else 0
                    categories[id] = {
                        'id': id,
                        'name': name,
                        'isthing': isthing
                    }
                self.categories = categories

            pq_stat = PQStat()
            for result in results:
                pq_stat += result

        metrics = [('All', None), ('Things', True), ('Stuff', False)]
        pq_results = {}

        for name, isthing in metrics:
            pq_results[name], classwise_results = pq_stat.pq_average(
                self.categories, isthing=isthing)
            if name == 'All':
                pq_results['classwise'] = classwise_results

        classwise_results = None
        if self.classwise:
            classwise_results = {
                k: v
                for k, v in zip(self.dataset_meta['classes'],
                                pq_results['classwise'].values())
            }

        self._print_panoptic_table(pq_results, classwise_results, logger=logger)
        results = self._parse_pq_results(pq_results)

        return results

    def _parse_pq_results(self, pq_results: dict) -> dict:
        """Parse the Panoptic Quality results.

        Args:
            pq_results (dict): Panoptic Quality results.

        Returns:
            dict: Panoptic Quality results parsed.
        """
        result = dict()
        result['PQ'] = 100 * pq_results['All']['pq']
        result['SQ'] = 100 * pq_results['All']['sq']
        result['RQ'] = 100 * pq_results['All']['rq']
        result['PQ_th'] = 100 * pq_results['Things']['pq']
        result['SQ_th'] = 100 * pq_results['Things']['sq']
        result['RQ_th'] = 100 * pq_results['Things']['rq']
        result['PQ_st'] = 100 * pq_results['Stuff']['pq']
        result['SQ_st'] = 100 * pq_results['Stuff']['sq']
        result['RQ_st'] = 100 * pq_results['Stuff']['rq']
        return result

    def _print_panoptic_table(
            self,
            pq_results: dict,
            classwise_results: Optional[dict] = None,
            logger: Optional[Union['MMLogger', str]] = None) -> None:
        """Print the panoptic evaluation results table.

        Args:
            pq_results(dict): The Panoptic Quality results.
            classwise_results(dict, optional): The classwise Panoptic Quality.
                results. The keys are class names and the values are metrics.
                Defaults to None.
            logger (:obj:`MMLogger` | str, optional): Logger used for printing
                related information during evaluation. Default: None.
        """

        headers = ['', 'PQ', 'SQ', 'RQ', 'categories']
        data = [headers]
        for name in ['All', 'Things', 'Stuff']:
            numbers = [
                f'{(pq_results[name][k] * 100):0.3f}' for k in ['pq', 'sq', 'rq']
            ]
            row = [name] + numbers + [pq_results[name]['n']]
            data.append(row)
        table = AsciiTable(data)
        print_log('ADE Panoptic Evaluation Results:\n' + table.table, logger=logger)

        if classwise_results is not None:
            class_metrics = [(name, ) + tuple(f'{(metrics[k] * 100):0.3f}'
                                              for k in ['pq', 'sq', 'rq'])
                             for name, metrics in classwise_results.items()]
            num_columns = min(8, len(class_metrics) * 4)
            results_flatten = list(itertools.chain(*class_metrics))
            headers = ['category', 'PQ', 'SQ', 'RQ'] * (num_columns // 4)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::num_columns] for i in range(num_columns)])
            data = [headers]
            data += [result for result in results_2d]
            table = AsciiTable(data)
            print_log(
                'Classwise ADE Panoptic Evaluation Results:\n' + table.table,
                logger=logger)
