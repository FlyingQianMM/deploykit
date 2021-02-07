# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from .base_preprocessing import BasePreprocessor
from deploykit.common.blob import DataBlob, ShapeInfo


class DetPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super(BasePreprocessor, self).__init__(config)
        self.architecture = self.config['model_name']
        self.build_transforms(self.config['transforms'])

    def __call__(self, img_list):
        shape_info_list = list()
        for img in img_list:
            height, width, _ = image.shape
            shape_info = ShapeInfo()
            shape_info.set_origin_shape(height, width)
            shape_info_list.append(shape_info)

        img_list = self.transform_image(img_list)
        shape_info_list = self.infer_shape(shape_info_list)

        im_data_blob = DataBlob()
        im_data_blob.name = 'image'
        im_data_blob.data = np.concatenate(img_list, axis=0)
        im_data_blob.shape = image.shape

        im_shape_blob = DataBlob()
        im_shape_blob.name = 'im_shape'
        im_shape_list = list()
        for shape_info in shape_info_list:
            origin_width, origin_height = shape_info.get_last_shape()
            im_shape_list.append(
                np.asarray(
                    (origin_height, origin_width, 1), dtype=np.float32))
        im_shape_blob.data = np.concatenate(im_shape_list, axis=0)

        inputs = list()
        if self.architecture in ['RCNN', 'RetinaNet']:
            im_info_blob = DataBlob()
            im_info_blob.name = 'im_info'
            im_info_list = list()
            for shape_info in shape_info_list:
                origin_width, origin_height = shape_info.get_last_shape()
                im_info = (origin_height, origin_width, 1)
                for op_name in list(shape_info.shape.keys()):
                    if 'Resize' in op_name:
                        resized_scale = float(origin_width) / float(
                            shape_info.shape[op_name][0])
                        im_info = shape_info.shape[op_name] + (resized_scale, )
                im_info_list.append(np.asarray(im_info, dtype=np.float32))
            im_info_blob.data = np.concatenate(im_info_list, axis=0)
            inputs.extend((im_data_blob, im_info_blob, im_shape_blob))
        elif self.architecture == ['YOLO', 'SSD']:
            inputs.extend((im_data_blob, im_shape_blob))
        return inputs, shape_info_list
