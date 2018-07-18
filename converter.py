import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
import urllib
import yaml
import shutil

from enum import Enum

from common import CaffeEnvType
from common import DeviceType
from common import mace_check
from common import MaceLogger
from common import StringFormatter

from mace.python.tools.converter import mace_convert_model

BUILD_OUTPUT_DIR = 'builds'
BUILD_DOWNLOADS_DIR = BUILD_OUTPUT_DIR + '/downloads'
PHONE_DATA_DIR = "/data/local/tmp/mace_run"
MODEL_OUTPUT_DIR_NAME = 'model'
MODEL_HEADER_DIR_PATH = 'include/mace/public'
BUILD_TMP_DIR_NAME = '_tmp'
BUILD_TMP_GENERAL_OUTPUT_DIR_NAME = 'general'
OUTPUT_LIBRARY_DIR_NAME = 'lib'
OUTPUT_OPENCL_BINARY_DIR_NAME = 'opencl'
OUTPUT_OPENCL_BINARY_FILE_NAME = 'compiled_opencl_kernel'
OUTPUT_OPENCL_PARAMETER_FILE_NAME = 'tuned_opencl_parameter'
CL_COMPILED_BINARY_FILE_NAME = "mace_cl_compiled_program.bin"
CL_TUNED_PARAMETER_FILE_NAME = "mace_run.config"
CODEGEN_BASE_DIR = 'mace/codegen'
MODEL_CODEGEN_DIR = CODEGEN_BASE_DIR + '/models'
ENGINE_CODEGEN_DIR = CODEGEN_BASE_DIR + '/engine'
LIB_CODEGEN_DIR = CODEGEN_BASE_DIR + '/lib'
LIBMACE_SO_TARGET = "//mace/libmace:libmace.so"
LIBMACE_STATIC_TARGET = "//mace/libmace:libmace_static"
LIBMACE_STATIC_PATH = "bazel-genfiles/mace/libmace/libmace.a"
LIBMACE_DYNAMIC_PATH = "bazel-bin/mace/libmace/libmace.so"
MODEL_LIB_TARGET = "//mace/codegen:generated_models"
MODEL_LIB_PATH = "bazel-bin/mace/codegen/libgenerated_models.a"
MACE_RUN_STATIC_NAME = "mace_run_static"
MACE_RUN_DYNAMIC_NAME = "mace_run_dynamic"
MACE_RUN_STATIC_TARGET = "//mace/tools/validation:" + MACE_RUN_STATIC_NAME
MACE_RUN_DYNAMIC_TARGET = "//mace/tools/validation:" + MACE_RUN_DYNAMIC_NAME
EXAMPLE_STATIC_NAME = "example_static"
EXAMPLE_DYNAMIC_NAME = "example_dynamic"
EXAMPLE_STATIC_TARGET = "//mace/examples/cli:" + EXAMPLE_STATIC_NAME
EXAMPLE_DYNAMIC_TARGET = "//mace/examples/cli:" + EXAMPLE_DYNAMIC_NAME
BM_MODEL_STATIC_NAME = "benchmark_model_static"
BM_MODEL_DYNAMIC_NAME = "benchmark_model_dynamic"
BM_MODEL_STATIC_TARGET = "//mace/benchmark:" + BM_MODEL_STATIC_NAME
BM_MODEL_DYNAMIC_TARGET = "//mace/benchmark:" + BM_MODEL_DYNAMIC_NAME
DEVICE_INTERIOR_DIR = PHONE_DATA_DIR + "/interior"
BUILD_TMP_OPENCL_BIN_DIR = 'opencl_bin'
ALL_SOC_TAG = 'all'

ABITypeStrs = [
    'armeabi-v7a',
    'arm64-v8a',
    'host',
]


class ABIType(object):
    armeabi_v7a = 'armeabi-v7a'
    arm64_v8a = 'arm64-v8a'
    host = 'host'


ModelFormatStrs = [
    "file",
    "code",
]


class MACELibType(object):
    static = 0
    dynamic = 1


PlatformTypeStrs = [
    "tensorflow",
    "caffe",
]
PlatformType = Enum('PlatformType', [(ele, ele) for ele in PlatformTypeStrs],
                    type=str)

RuntimeTypeStrs = [
    "cpu",
    "gpu",
    "dsp",
    "cpu+gpu"
]

class RuntimeType(object):
    cpu = 'cpu'
    gpu = 'gpu'
    dsp = 'dsp'
    cpu_gpu = 'cpu+gpu'


CPUDataTypeStrs = [
    "fp32",
]

CPUDataType = Enum('CPUDataType', [(ele, ele) for ele in CPUDataTypeStrs],
                   type=str)

GPUDataTypeStrs = [
    "fp16_fp32",
    "fp32_fp32",
]

GPUDataType = Enum('GPUDataType', [(ele, ele) for ele in GPUDataTypeStrs],
                   type=str)

DSPDataTypeStrs = [
    "uint8",
]

DSPDataType = Enum('DSPDataType', [(ele, ele) for ele in DSPDataTypeStrs],
                   type=str)

WinogradParameters = [0, 2, 4]

class BuildType(object):
    proto = 'proto'
    code = 'code'


class ModelFormat(object):
    file = 'file'
    code = 'code'

class DefaultValues(object):
    mace_lib_type = MACELibType.static
    omp_num_threads = -1,
    cpu_affinity_policy = 1,
    gpu_perf_hint = 3,
    gpu_priority_hint = 3,


class YAMLKeyword(object):
    library_name = 'library_name'
    target_abis = 'target_abis'
    target_socs = 'target_socs'
    model_graph_format = 'model_graph_format'
    model_data_format = 'model_data_format'
    models = 'models'
    platform = 'platform'
    model_file_path = 'model_file_path'
    model_sha256_checksum = 'model_sha256_checksum'
    weight_file_path = 'weight_file_path'
    weight_sha256_checksum = 'weight_sha256_checksum'
    subgraphs = 'subgraphs'
    input_tensors = 'input_tensors'
    input_shapes = 'input_shapes'
    input_ranges = 'input_ranges'
    output_tensors = 'output_tensors'
    output_shapes = 'output_shapes'
    runtime = 'runtime'
    data_type = 'data_type'
    limit_opencl_kernel_time = 'limit_opencl_kernel_time'
    nnlib_graph_mode = 'nnlib_graph_mode'
    obfuscate = 'obfuscate'
    winograd = 'winograd'
    validation_inputs_data = 'validation_inputs_data'
    graph_optimize_options = 'graph_optimize_options'  # internal use for now


class ModuleName(object):
    YAML_CONFIG = 'YAML CONFIG'
    MODEL_CONVERTER = 'Model Converter'
    RUN = 'RUN'
    BENCHMARK = 'Benchmark'


CPP_KEYWORDS = [
    'alignas', 'alignof', 'and', 'and_eq', 'asm', 'atomic_cancel',
    'atomic_commit', 'atomic_noexcept', 'auto', 'bitand', 'bitor',
    'bool', 'break', 'case', 'catch', 'char', 'char16_t', 'char32_t',
    'class', 'compl', 'concept', 'const', 'constexpr', 'const_cast',
    'continue', 'co_await', 'co_return', 'co_yield', 'decltype', 'default',
    'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit',
    'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if',
    'import', 'inline', 'int', 'long', 'module', 'mutable', 'namespace',
    'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq',
    'private', 'protected', 'public', 'register', 'reinterpret_cast',
    'requires', 'return', 'short', 'signed', 'sizeof', 'static',
    'static_assert', 'static_cast', 'struct', 'switch', 'synchronized',
    'template', 'this', 'thread_local', 'throw', 'true', 'try', 'typedef',
    'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void',
    'volatile', 'wchar_t', 'while', 'xor', 'xor_eq', 'override', 'final',
    'transaction_safe', 'transaction_safe_dynamic', 'if', 'elif', 'else',
    'endif', 'defined', 'ifdef', 'ifndef', 'define', 'undef', 'include',
    'line', 'error', 'pragma',
]


################################
# common functions
################################
def parse_device_type(runtime):
    device_type = ""

    if runtime == RuntimeType.dsp:
        device_type = DeviceType.HEXAGON
    elif runtime == RuntimeType.gpu:
        device_type = DeviceType.GPU
    elif runtime == RuntimeType.cpu:
        device_type = DeviceType.CPU

    return device_type


def get_hexagon_mode(configs):
    runtime_list = []
    for model_name in configs[YAMLKeyword.models]:
        model_runtime =\
            configs[YAMLKeyword.models][model_name].get(
                YAMLKeyword.runtime, "")
        runtime_list.append(model_runtime.lower())

    if RuntimeType.dsp in runtime_list:
        return True
    return False


def get_opencl_mode(configs):
    runtime_list = []
    for model_name in configs[YAMLKeyword.models]:
        model_runtime =\
            configs[YAMLKeyword.models][model_name].get(
                YAMLKeyword.runtime, "")
        runtime_list.append(model_runtime.lower())

    if RuntimeType.gpu in runtime_list or RuntimeType.cpu_gpu in runtime_list:
        return True
    return False


def md5sum(str):
    md5 = hashlib.md5()
    md5.update(str)
    return md5.hexdigest()


def sha256_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def print_library_summary(configs):
    library_name = configs[YAMLKeyword.library_name]
    title = "Library"
    header = ["key", "value"]
    data = list()
    data.append(["MACE Model Path",
                 "%s/%s/%s"
                 % (BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME)])
    if configs[YAMLKeyword.model_graph_format] == ModelFormat.code:
        data.append(["MACE Model Header Path",
                     "%s/%s/%s"
                     % (BUILD_OUTPUT_DIR, library_name,
                        MODEL_HEADER_DIR_PATH)])

    MaceLogger.summary(StringFormatter.table(header, data, title))

def format_model_config(flags):
    with open(flags.config) as f:
        configs = yaml.load(f)

    library_name = configs.get(YAMLKeyword.library_name, "")
    mace_check(len(library_name) > 0,
               ModuleName.YAML_CONFIG, "library name should not be empty")

    if flags.target_abis:
        target_abis = flags.target_abis.split(',')
    else:
        target_abis = configs.get(YAMLKeyword.target_abis, [])
    mace_check((isinstance(target_abis, list) and len(target_abis) > 0),
               ModuleName.YAML_CONFIG, "target_abis list is needed")
    configs[YAMLKeyword.target_abis] = target_abis
    for abi in target_abis:
        mace_check(abi in ABITypeStrs,
                   ModuleName.YAML_CONFIG,
                   "target_abis must be in " + str(ABITypeStrs))

    target_socs = configs.get(YAMLKeyword.target_socs, "")
    if flags.target_socs:
        configs[YAMLKeyword.target_socs] = \
               [soc.lower() for soc in flags.target_socs.split(',')]
    elif not target_socs:
        configs[YAMLKeyword.target_socs] = []
    elif not isinstance(target_socs, list):
        configs[YAMLKeyword.target_socs] = [target_socs]

    configs[YAMLKeyword.target_socs] = \
        [soc.lower() for soc in configs[YAMLKeyword.target_socs]]

    if ABIType.armeabi_v7a in target_abis \
            or ABIType.arm64_v8a in target_abis:
        available_socs = set([])
        target_socs = configs[YAMLKeyword.target_socs]
        if ALL_SOC_TAG in target_socs:
            mace_check(available_socs,
                       ModuleName.YAML_CONFIG,
                       "Build for all SOCs plugged in computer, "
                       "you at least plug in one phone")
        else:
            for soc in target_socs:
                mace_check(soc in available_socs,
                           ModuleName.YAML_CONFIG,
                           "Build specified SOC library, "
                           "you must plug in a phone using the SOC")

    if flags.model_graph_format:
        model_graph_format = flags.model_graph_format
    else:
        model_graph_format = configs.get(YAMLKeyword.model_graph_format, "")
    mace_check(model_graph_format in ModelFormatStrs,
               ModuleName.YAML_CONFIG,
               'You must set model_graph_format and '
               "model_graph_format must be in " + str(ModelFormatStrs))
    configs[YAMLKeyword.model_graph_format] = model_graph_format
    if flags.model_data_format:
        model_data_format = flags.model_data_format
    else:
        model_data_format = configs.get(YAMLKeyword.model_data_format, "")
    configs[YAMLKeyword.model_data_format] = model_data_format
    mace_check(model_data_format in ModelFormatStrs,
               ModuleName.YAML_CONFIG,
               'You must set model_data_format and '
               "model_data_format must be in " + str(ModelFormatStrs))

    mace_check(not (model_graph_format == ModelFormat.file
                    and model_data_format == ModelFormat.code),
               ModuleName.YAML_CONFIG,
               "If model_graph format is 'file',"
               " the model_data_format must be 'file' too")

    model_names = configs.get(YAMLKeyword.models, [])
    mace_check(len(model_names) > 0, ModuleName.YAML_CONFIG,
               "no model found in config file")

    model_name_reg = re.compile(r'^[a-zA-Z0-9_]+$')
    for model_name in model_names:
        # check model_name legality
        mace_check(model_name not in CPP_KEYWORDS,
                   ModuleName.YAML_CONFIG,
                   "model name should not be c++ keyword.")
        mace_check((model_name[0] == '_' or model_name[0].isalpha())
                   and bool(model_name_reg.match(model_name)),
                   ModuleName.YAML_CONFIG,
                   "model name should Meet the c++ naming convention"
                   " which start with '_' or alpha"
                   " and only contain alpha, number and '_'")

        model_config = configs[YAMLKeyword.models][model_name]
        platform = model_config.get(YAMLKeyword.platform, "")
        mace_check(platform in PlatformTypeStrs,
                   ModuleName.YAML_CONFIG,
                   "'platform' must be in " + str(PlatformTypeStrs))

        for key in [YAMLKeyword.model_file_path]:
            value = model_config.get(key, "")
            mace_check(value != "", ModuleName.YAML_CONFIG, "'%s' is necessary" % key)

        weight_file_path = model_config.get(YAMLKeyword.weight_file_path, "")
        if weight_file_path:
            pass
        else:
            model_config[YAMLKeyword.weight_sha256_checksum] = ""

        runtime = model_config.get(YAMLKeyword.runtime, "")
        mace_check(runtime in RuntimeTypeStrs,
                   ModuleName.YAML_CONFIG,
                   "'runtime' must be in " + str(RuntimeTypeStrs))
        if ABIType.host in target_abis:
            mace_check(runtime == RuntimeType.cpu,
                       ModuleName.YAML_CONFIG,
                       "host only support cpu runtime now.")

        data_type = model_config.get(YAMLKeyword.data_type, "")
        if runtime == RuntimeType.cpu_gpu and data_type not in GPUDataTypeStrs:
            model_config[YAMLKeyword.data_type] = \
                GPUDataType.fp16_fp32.value
        elif runtime == RuntimeType.cpu:
            if len(data_type) > 0:
                mace_check(data_type in CPUDataTypeStrs,
                           ModuleName.YAML_CONFIG,
                           "'data_type' must be in " + str(CPUDataTypeStrs)
                           + " for cpu runtime")
            else:
                model_config[YAMLKeyword.data_type] = \
                    CPUDataType.fp32.value
        elif runtime == RuntimeType.gpu:
            if len(data_type) > 0:
                mace_check(data_type in GPUDataTypeStrs,
                           ModuleName.YAML_CONFIG,
                           "'data_type' must be in " + str(GPUDataTypeStrs)
                           + " for gpu runtime")
            else:
                model_config[YAMLKeyword.data_type] =\
                    GPUDataType.fp16_fp32.value
        elif runtime == RuntimeType.dsp:
            if len(data_type) > 0:
                mace_check(data_type in DSPDataTypeStrs,
                           ModuleName.YAML_CONFIG,
                           "'data_type' must be in " + str(DSPDataTypeStrs)
                           + " for dsp runtime")
            else:
                model_config[YAMLKeyword.data_type] = \
                    DSPDataType.uint8.value

        subgraphs = model_config.get(YAMLKeyword.subgraphs, "")
        mace_check(len(subgraphs) > 0, ModuleName.YAML_CONFIG,
                   "at least one subgraph is needed")

        for subgraph in subgraphs:
            for key in [YAMLKeyword.input_tensors,
                        YAMLKeyword.input_shapes,
                        YAMLKeyword.output_tensors,
                        YAMLKeyword.output_shapes]:
                value = subgraph.get(key, "")
                mace_check(value != "", ModuleName.YAML_CONFIG,
                           "'%s' is necessary in subgraph" % key)
                if not isinstance(value, list):
                    subgraph[key] = [value]
            validation_inputs_data = subgraph.get(
                YAMLKeyword.validation_inputs_data, [])
            if not isinstance(validation_inputs_data, list):
                subgraph[YAMLKeyword.validation_inputs_data] = [
                    validation_inputs_data]
            else:
                subgraph[YAMLKeyword.validation_inputs_data] = \
                    validation_inputs_data
            input_ranges = subgraph.get(
                YAMLKeyword.input_ranges, [])
            if not isinstance(input_ranges, list):
                subgraph[YAMLKeyword.input_ranges] = [input_ranges]
            else:
                subgraph[YAMLKeyword.input_ranges] = input_ranges

        for key in [YAMLKeyword.limit_opencl_kernel_time,
                    YAMLKeyword.nnlib_graph_mode,
                    YAMLKeyword.obfuscate,
                    YAMLKeyword.winograd]:
            value = model_config.get(key, "")
            if value == "":
                model_config[key] = 0

        mace_check(model_config[YAMLKeyword.winograd] in WinogradParameters,
                   ModuleName.YAML_CONFIG,
                   "'winograd' parameters must be in "
                   + str(WinogradParameters) +
                   ". 0 for disable winograd convolution")

        weight_file_path = model_config.get(YAMLKeyword.weight_file_path, "")
        model_config[YAMLKeyword.weight_file_path] = weight_file_path

    return configs

################################
# convert
################################
def print_configuration(configs):
    title = "Common Configuration"
    header = ["key", "value"]
    data = list()
    data.append([YAMLKeyword.library_name,
                 configs[YAMLKeyword.library_name]])
    data.append([YAMLKeyword.target_abis,
                 configs[YAMLKeyword.target_abis]])
    data.append([YAMLKeyword.target_socs,
                 configs[YAMLKeyword.target_socs]])
    data.append([YAMLKeyword.model_graph_format,
                 configs[YAMLKeyword.model_graph_format]])
    data.append([YAMLKeyword.model_data_format,
                 configs[YAMLKeyword.model_data_format]])
    MaceLogger.summary(StringFormatter.table(header, data, title))


def get_model_files(model_file_path,
                    model_sha256_checksum,
                    model_output_dir,
                    weight_file_path="",
                    weight_sha256_checksum=""):
    model_file = model_file_path
    weight_file = weight_file_path

    # if model_file_path.startswith("http://") or \
    #         model_file_path.startswith("https://"):
    #     model_file = model_output_dir + "/" + md5sum(model_file_path) + ".pb"
    #     if not os.path.exists(model_file) or \
    #             sha256_checksum(model_file) != model_sha256_checksum:
    #         MaceLogger.info("Downloading model, please wait ...")
    #         urllib.urlretrieve(model_file_path, model_file)
    #         MaceLogger.info("Model downloaded successfully.")
    #
    # if sha256_checksum(model_file) != model_sha256_checksum:
    #     MaceLogger.error(ModuleName.MODEL_CONVERTER,
    #                      "model file sha256checksum not match")
    #
    # if weight_file_path.startswith("http://") or \
    #         weight_file_path.startswith("https://"):
    #     weight_file = \
    #         model_output_dir + "/" + md5sum(weight_file_path) + ".caffemodel"
    #     if not os.path.exists(weight_file) or \
    #             sha256_checksum(weight_file) != weight_sha256_checksum:
    #         MaceLogger.info("Downloading model weight, please wait ...")
    #         urllib.urlretrieve(weight_file_path, weight_file)
    #         MaceLogger.info("Model weight downloaded successfully.")
    #
    # if weight_file:
    #     if sha256_checksum(weight_file) != weight_sha256_checksum:
    #         MaceLogger.error(ModuleName.MODEL_CONVERTER,
    #                          "weight file sha256checksum not match")

    return model_file, weight_file

def gen_model_code(model_codegen_dir,
                   platform,
                   model_file_path,
                   weight_file_path,
                   model_sha256_checksum,
                   weight_sha256_checksum,
                   input_nodes,
                   output_nodes,
                   runtime,
                   model_tag,
                   input_shapes,
                   dsp_mode,
                   embed_model_data,
                   winograd,
                   obfuscate,
                   model_graph_format,
                   data_type,
                   graph_optimize_options):

    if os.path.exists(model_codegen_dir):
        # sh.rm("-rf", model_codegen_dir)
        shutil.rmtree(model_codegen_dir)
    # sh.mkdir("-p", model_codegen_dir)
    os.makedirs(model_codegen_dir)

    # mace_convert_model(platform, model_file_path,
    #           "--weight_file=%s " % weight_file_path,
    #           "--model_checksum=%s " % model_sha256_checksum,
    #           "--weight_checksum=%s " % weight_sha256_checksum,
    #           "--input_node=%s " % input_nodes,
    #           "--output_node=%s " % output_nodes,
    #           "--runtime=%s " % runtime,
    #           "--template=%s " % "mace/python/tools",
    #           "--model_tag=%s " % model_tag,
    #           "--input_shape=%s " % input_shapes,
    #           "--dsp_mode=%s " % dsp_mode,
    #           "--embed_model_data=%s " % embed_model_data,
    #           "--winograd=%s " % winograd,
    #           "--obfuscate=%s " % obfuscate,
    #           "--output_dir=%s " % model_codegen_dir,
    #           "--model_graph_format=%s " % model_graph_format,
    #           "--data_type=%s " % data_type,
    #           "--graph_optimize_options=%s " % graph_optimize_options)
    mace_convert_model(platform,
                       model_file_path,
                       model_sha256_checksum,
                       weight_file_path,
                       weight_sha256_checksum,
                       runtime,
                       data_type,
                       input_nodes,
                       input_shapes,
                       output_nodes,
                       dsp_mode,
                       graph_optimize_options,
                       winograd,
                       "mace/python/tools",
                       obfuscate,
                       model_tag,
                       model_codegen_dir,
                       embed_model_data,
                       model_graph_format)

def convert_model(configs):
    # Remove previous output dirs
    library_name = configs[YAMLKeyword.library_name]
    if not os.path.exists(BUILD_OUTPUT_DIR):
        os.makedirs(BUILD_OUTPUT_DIR)
    elif os.path.exists(os.path.join(BUILD_OUTPUT_DIR, library_name)):
        # sh.rm("-rf", os.path.join(BUILD_OUTPUT_DIR, library_name))
        shutil.rmtree(os.path.join(BUILD_OUTPUT_DIR, library_name))
    os.mkdir(os.path.join(BUILD_OUTPUT_DIR, library_name))
    if not os.path.exists(BUILD_DOWNLOADS_DIR):
        os.mkdir(BUILD_DOWNLOADS_DIR)

    model_output_dir = \
        '%s/%s/%s' % (BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME)
    model_header_dir = \
        '%s/%s/%s' % (BUILD_OUTPUT_DIR, library_name, MODEL_HEADER_DIR_PATH)
    # clear output dir
    if os.path.exists(model_output_dir):
        # sh.rm("-rf", model_output_dir)
        shutil.rmtree(model_output_dir)
    os.makedirs(model_output_dir)
    if os.path.exists(model_header_dir):
        # sh.rm("-rf", model_header_dir)
        shutil.rmtree(model_header_dir)

    embed_model_data = \
        configs[YAMLKeyword.model_data_format] == ModelFormat.code

    if os.path.exists(MODEL_CODEGEN_DIR):
        # sh.rm("-rf", MODEL_CODEGEN_DIR)
        shutil.rmtree(MODEL_CODEGEN_DIR)
    if os.path.exists(ENGINE_CODEGEN_DIR):
        # sh.rm("-rf", ENGINE_CODEGEN_DIR)
        shutil.rmtree(ENGINE_CODEGEN_DIR)

    for model_name in configs[YAMLKeyword.models]:
        MaceLogger.header(
            StringFormatter.block("Convert %s model" % model_name))
        model_config = configs[YAMLKeyword.models][model_name]
        runtime = model_config[YAMLKeyword.runtime]

        model_file_path, weight_file_path = get_model_files(
            model_config[YAMLKeyword.model_file_path],
            model_config[YAMLKeyword.model_sha256_checksum],
            BUILD_DOWNLOADS_DIR,
            model_config[YAMLKeyword.weight_file_path],
            model_config[YAMLKeyword.weight_sha256_checksum])

        data_type = model_config[YAMLKeyword.data_type]
        # TODO(liuqi): support multiple subgraphs
        subgraphs = model_config[YAMLKeyword.subgraphs]

        model_codegen_dir = "%s/%s" % (MODEL_CODEGEN_DIR, model_name)
        gen_model_code(
            model_codegen_dir,
            model_config[YAMLKeyword.platform],
            model_file_path,
            weight_file_path,
            model_config[YAMLKeyword.model_sha256_checksum],
            model_config[YAMLKeyword.weight_sha256_checksum],
            ",".join(subgraphs[0][YAMLKeyword.input_tensors]),
            ",".join(subgraphs[0][YAMLKeyword.output_tensors]),
            runtime,
            model_name,
            ":".join(subgraphs[0][YAMLKeyword.input_shapes]),
            model_config[YAMLKeyword.nnlib_graph_mode],
            embed_model_data,
            model_config[YAMLKeyword.winograd],
            model_config[YAMLKeyword.obfuscate],
            configs[YAMLKeyword.model_graph_format],
            data_type,
            ",".join(model_config.get(YAMLKeyword.graph_optimize_options, [])))

        # if configs[YAMLKeyword.model_graph_format] == ModelFormat.file:
        #     sh.mv("-f",
        #           '%s/%s.pb' % (model_codegen_dir, model_name),
        #           model_output_dir)
        #     sh.mv("-f",
        #           '%s/%s.data' % (model_codegen_dir, model_name),
        #           model_output_dir)
        # else:
        #     if not embed_model_data:
        #         sh.mv("-f",
        #               '%s/%s.data' % (model_codegen_dir, model_name),
        #               model_output_dir)
        #     sh.cp("-f", glob.glob("mace/codegen/models/*/*.h"),
        #           model_header_dir)

        MaceLogger.summary(
            StringFormatter.block("Model %s converted" % model_name))

def convert_func(flags):
    configs = format_model_config(flags)

    print_configuration(configs)

    convert_model(configs)

    print_library_summary(configs)

def parse_args():
    """Parses command line arguments."""
    all_type_parent_parser = argparse.ArgumentParser(add_help=False)
    all_type_parent_parser.add_argument(
        '--config',
        type=str,
        default="",
        required=True,
        help="the path of model yaml configuration file.")
    all_type_parent_parser.add_argument(
        "--model_graph_format",
        type=str,
        default="",
        help="[file, code], MACE Model graph format.")
    all_type_parent_parser.add_argument(
        "--model_data_format",
        type=str,
        default="",
        help="['file', 'code'], MACE Model data format.")
    all_type_parent_parser.add_argument(
        "--target_abis",
        type=str,
        default="",
        help="Target ABIs, comma seperated list.")
    all_type_parent_parser.add_argument(
        "--target_socs",
        type=str,
        default="",
        help="Target SOCs, comma seperated list.")
    convert_run_parent_parser = argparse.ArgumentParser(add_help=False)
    convert_run_parent_parser.add_argument(
        '--address_sanitizer',
        action="store_true",
        help="Whether to use address sanitizer to check memory error")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    convert = subparsers.add_parser(
        'convert',
        parents=[all_type_parent_parser, convert_run_parent_parser],
        help='convert to mace model (file or code)')
    convert.set_defaults(func=convert_func)
    return parser.parse_known_args()

if __name__ == "__main__":
    flags, unparsed = parse_args()
    flags.func(flags)