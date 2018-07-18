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

import enum
import re

################################
# log
################################
class CMDColors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MaceLogger:
    @staticmethod
    def header(message):
        print(CMDColors.PURPLE + message + CMDColors.ENDC)

    @staticmethod
    def summary(message):
        print(CMDColors.GREEN + message + CMDColors.ENDC)

    @staticmethod
    def info(message):
        print(message)

    @staticmethod
    def warning(message):
        print(CMDColors.YELLOW + 'WARNING:' + message + CMDColors.ENDC)

    @staticmethod
    def error(module, message):
        print(CMDColors.RED + 'ERROR: [' + module + '] '+ message + CMDColors.ENDC)
        exit(1)


def mace_check(condition, module, message):
    if not condition:
        MaceLogger.error(module, message)


################################
# String Formatter
################################
class StringFormatter:
    @staticmethod
    def table(header, data, title, align="R"):
        data_size = len(data)
        column_size = len(header)
        column_length = [len(str(ele)) + 1 for ele in header]
        for row_idx in range(data_size):
            data_tuple = data[row_idx]
            ele_size = len(data_tuple)
            assert(ele_size == column_size)
            for i in range(ele_size):
                column_length[i] = max(column_length[i],
                                       len(str(data_tuple[i])) + 1)

        table_column_length = sum(column_length) + column_size + 1
        dash_line = '-' * table_column_length + '\n'
        header_line = '=' * table_column_length + '\n'
        output = ""
        output += dash_line
        output += str(title).center(table_column_length) + '\n'
        output += dash_line
        output += '|' + '|'.join([str(header[i]).center(column_length[i])
                                  for i in range(column_size)]) + '|\n'
        output += header_line

        for data_tuple in data:
            ele_size = len(data_tuple)
            row_list = []
            for i in range(ele_size):
                if align == "R":
                    row_list.append(str(data_tuple[i]).rjust(column_length[i]))
                elif align == "L":
                    row_list.append(str(data_tuple[i]).ljust(column_length[i]))
                elif align == "C":
                    row_list.append(str(data_tuple[i])
                                    .center(column_length[i]))
            output += '|' + '|'.join(row_list) + "|\n" + dash_line
        return output

    @staticmethod
    def block(message):
        line_length = 10 + len(str(message)) + 10
        star_line = '*' * line_length + '\n'
        return star_line + str(message).center(line_length) + '\n' + star_line


################################
# definitions
################################
class DeviceType(object):
    CPU = 'CPU'
    GPU = 'GPU'
    HEXAGON = 'HEXAGON'


################################
# Argument types
################################
class CaffeEnvType(enum.Enum):
    DOCKER = 0,
    LOCAL = 1,


################################
# common functions
################################
def formatted_file_name(input_file_name, input_name):
    res = input_file_name + '_'
    for c in input_name:
        res += c if c.isalnum() else '_'
    return res
