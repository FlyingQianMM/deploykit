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

import os
import os.path as osp
import yaml
from collections import OrderedDict


def parse_ppdet_config(cfg_file):
    with open(cfg_file) as f:
        yaml_dict = yaml.load(f.read(), Loader=yaml.Loader)

    config = OrderedDict()
    config["model_format"] = "Paddle"
    if "arch" not in yaml_dict:
        raise Exception(
            "Failed to find `arch` in the recieved PaddleDetection yaml file, please check it should be set as one of YOLO/SSD/RetinaNet/EfficientDet/RCNN/Face/TTF/FCOS/SOLOv2."
        )
    config["model_name"] = yaml_dict["arch"]
    config["toolkit"] = "PaddleDetection"
    config["toolkit_version"] = "Unknown"
    if "label_list" not in yaml_dict:
        raise Exception(
            "Failed to find `label_list` in the recieved PaddleDetection yaml file."
        )
    config["lables"] = yaml_dict["label_list"]

    # Parse preprocess in paddledetection yaml file
    if "Preprocess" not in yaml_dict:
        raise Exception(
            "Failed to find `Preprocess` in the recieved PaddleDetection yaml file."
        )
    preprocess_info = yaml_dict['Preprocess']
    config["transforms"] = parse_preprocess(preprocess_info)
    return config


def parse_preprocess(preprocess_info):
    transforms = OrderedDict()
    for op_info in preprocess_info:
        op_name = op_info["type"]
        if op_name == "Normalize":
            transforms["Normalize"] = dict()
            transforms["Normalize"]["mean"] = op_info["mean"]
            transforms["Normalize"]["std"] = op_info["std"]
            transforms["Normalize"]["min_val"] = (0, ) * len(op_info["mean"])
            transforms["Normalize"]["max_val"] = (
                255., ) * len(op_info["mean"])
            transforms["Normalize"]["is_scale"] = op_info["is_scale"]
        elif op_name == "Permute":
            transforms["Permute"] = []
            if op_info["to_bgr"]:
                transforms["RGB2BRG"] = []
        elif op_name == "Resize":
            max_size = op_info["max_size"]
            if max_size != 0 and config["model_name"] in ["RCNN", "RetinaNet"]:
                transforms["ResizeByShort"] = dict()
                transforms["ResizeByShort"]["target_size"] = op_info[
                    "target_size"]
                transforms["ResizeByShort"]["max_size"] = op_info["max_size"]
                transforms["ResizeByShort"]["interp"] = op_info["interp"]
                if "image_shape" in op_info:
                    transforms["ResizeByShort"]["image_shape"] = op_info[
                        "image_shape"]
            else:
                transforms["Resize"] = dict()
                transforms["Resize"]["width"] = op_info["target_size"]
                transforms["Resize"]["height"] = op_info["target_size"]
                transforms["Resize"]["max_size"] = op_info["max_size"]
                transforms["Resize"]["interp"] = op_info["interp"]
        elif op_name == "PadStride":
            transforms["Padding"] = dict()
            transforms["Padding"]["stride"] = op_info["stride"]
        else:
            raise Exception("Cannot parse the operation {}.".format(op_name))
    return transforms
