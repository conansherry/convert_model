# Copyright 2018 Xiaomi, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import hashlib
import os.path
import copy

from mace.proto import mace_pb2
from mace.python.tools import memory_optimizer
from mace.python.tools import model_saver
from mace.python.tools.converter_tool import base_converter as cvt
from mace.python.tools.converter_tool import transformer
from mace.python.tools.convert_util import mace_check


# ./bazel-bin/mace/python/tools/tf_converter --model_file quantized_test.pb \
#                                            --output quantized_test_dsp.pb \
#                                            --runtime dsp \
#                                            --input_dim input_node,1,28,28,3

FLAGS = None

device_type_map = {'cpu': cvt.DeviceType.CPU.value,
                   'gpu': cvt.DeviceType.GPU.value,
                   'dsp': cvt.DeviceType.HEXAGON.value}


def parse_data_type(data_type, device_type):
    if device_type == cvt.DeviceType.GPU.value:
        if data_type == 'fp32_fp32':
            return mace_pb2.DT_FLOAT
        else:
            return mace_pb2.DT_HALF
    elif device_type == cvt.DeviceType.CPU.value:
        return mace_pb2.DT_FLOAT
    elif device_type == cvt.DeviceType.HEXAGON.value:
        return mace_pb2.DT_UINT8
    else:
        print("Invalid device type: " + device_type)


def file_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def parse_int_array_from_str(ints_str):
    return [int(int_str) for int_str in ints_str.split(',')]


def mace_convert_model(platform,
                       model_file,
                       model_checksum_in,
                       weight_file,
                       weight_checksum_in,
                       runtime,
                       data_type,
                       input_node,
                       input_shape,
                       output_node,
                       dsp_mode,
                       graph_optimize_options,
                       winograd,
                       template_dir,
                       obfuscate,
                       model_tag,
                       output_dir,
                       embed_model_data,
                       model_graph_format):
    if not os.path.isfile(model_file):
        print("Input graph file '" + model_file + "' does not exist!")
        sys.exit(-1)

    model_checksum = file_checksum(model_file)
    if model_checksum_in is not None and model_checksum_in != model_checksum:
        print("Model checksum mismatch: %s != %s" % (model_checksum,
                                                     model_checksum_in))
        sys.exit(-1)

    weight_checksum = None
    if platform == 'caffe':
        if not os.path.isfile(weight_file):
            print("Input weight file '" + weight_file +
                  "' does not exist!")
            sys.exit(-1)

        weight_checksum = file_checksum(weight_file)
        if weight_checksum_in is not None and \
                        weight_checksum_in != weight_checksum:
            print("Weight checksum mismatch: %s != %s" %
                  (weight_checksum, weight_checksum_in))
            sys.exit(-1)

    if platform not in ['tensorflow', 'caffe']:
        print("platform %s is not supported." % platform)
        sys.exit(-1)
    if runtime not in ['cpu', 'gpu', 'dsp', 'cpu+gpu']:
        print("runtime %s is not supported." % runtime)
        sys.exit(-1)

    if runtime == 'dsp':
        if platform == 'tensorflow':
            from mace.python.tools import tf_dsp_converter_lib
            output_graph_def = tf_dsp_converter_lib.convert_to_mace_pb(
                model_file, input_node, output_node,
                dsp_mode)
        else:
            print("%s does not support dsp runtime yet." % platform)
            sys.exit(-1)
    else:
        if graph_optimize_options:
            option = cvt.ConverterOption(
                graph_optimize_options.split(','))
        else:
            option = cvt.ConverterOption()
        option.winograd = winograd

        input_node_names = input_node.split(',')
        input_node_shapes = input_shape.split(':')
        if len(input_node_names) != len(input_node_shapes):
            raise Exception('input node count and shape count do not match.')
        for i in range(len(input_node_names)):
            input_node = cvt.NodeInfo()
            input_node.name = input_node_names[i]
            input_node.shape = parse_int_array_from_str(input_node_shapes[i])
            option.add_input_node(input_node)

        output_node_names = output_node.split(',')
        for i in range(len(output_node_names)):
            output_node = cvt.NodeInfo()
            output_node.name = output_node_names[i]
            option.add_output_node(output_node)

        if platform == 'tensorflow':
            from mace.python.tools.converter_tool import tensorflow_converter
            converter = tensorflow_converter.TensorflowConverter(
                option, model_file)
        elif platform == 'caffe':
            from mace.python.tools.converter_tool import caffe_converter
            converter = caffe_converter.CaffeConverter(option,
                                                       model_file,
                                                       weight_file)
        else:
            print("Mace do not support platorm %s yet." % platform)
            exit(1)

        output_graph_def = converter.run()

        print("Transform model to one that can better run on device")
        if runtime == 'cpu+gpu':
            cpu_graph_def = copy.deepcopy(output_graph_def)

            option.device = cvt.DeviceType.GPU.value
            option.data_type = parse_data_type(
                data_type, cvt.DeviceType.GPU.value)
            mace_gpu_transformer = transformer.Transformer(
                option, output_graph_def)
            output_graph_def = mace_gpu_transformer.run()
            print("start optimize gpu memory.")
            memory_optimizer.optimize_gpu_memory(output_graph_def)
            print("GPU memory optimization done.")

            option.device = cvt.DeviceType.CPU.value
            option.data_type = parse_data_type(
                data_type, cvt.DeviceType.CPU.value)
            option.disable_transpose_filters()
            mace_cpu_transformer = transformer.Transformer(
                option, cpu_graph_def)
            cpu_graph_def = mace_cpu_transformer.run()
            print("start optimize cpu memory.")
            memory_optimizer.optimize_cpu_memory(cpu_graph_def)
            print("CPU memory optimization done.")

            print("Merge cpu and gpu ops together")
            output_graph_def.op.extend(cpu_graph_def.op)
            output_graph_def.mem_arena.mem_block.extend(
                cpu_graph_def.mem_arena.mem_block)
            print("Merge done")
        else:
            option.device = device_type_map[runtime]
            option.data_type = parse_data_type(
                data_type, option.device)
            mace_transformer = transformer.Transformer(
                option, output_graph_def)
            output_graph_def = mace_transformer.run()

            print("start optimize memory.")
            if runtime == 'gpu':
                memory_optimizer.optimize_gpu_memory(output_graph_def)
            elif runtime == 'cpu':
                memory_optimizer.optimize_cpu_memory(output_graph_def)
            else:
                mace_check(False, "runtime only support [gpu|cpu|dsp]")

            print("Memory optimization done.")

    model_saver.save_model(
        output_graph_def, model_checksum, weight_checksum,
        template_dir, obfuscate, model_tag,
        output_dir, runtime,
        embed_model_data,
        winograd, data_type,
        model_graph_format)