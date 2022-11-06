import datetime
import re

import numpy as np
import pandas as pd
from dataclasses import dataclass
from dateutil.parser import *
from dateutil.parser import parse


@dataclass
class SystemControlData:
    data_tag: int = None
    byte_size: int = None
    continuation_flag: int = None
    start_offset: int = None
    data: bytes = None


def read_control_file(file_name, chunk_byte_size):
    system_control_data_list = list()
    with open(file_name, "rb") as control_file:
        while True:
            data_tag = control_file.read(chunk_byte_size)
            if data_tag:
                chunk = SystemControlData()
                chunk.data_tag = int.from_bytes(data_tag, 'little', signed=True)
                chunk.byte_size = int.from_bytes(control_file.read(chunk_byte_size), 'little', signed=True)
                chunk.continuation_flag = int.from_bytes(control_file.read(chunk_byte_size), 'little', signed=True)
                chunk.start_offset = int.from_bytes(control_file.read(chunk_byte_size), 'little', signed=True)
                chunk.data = control_file.read(chunk.byte_size)
                system_control_data_list.append(chunk)
            else:
                break
    return system_control_data_list


def is_date(ds):
    if re.match(r'\d{2}/\d{2}/\d{4}', ds):
        return bool(parse(ds))
    return False


def is_age(ds):
    if re.match(r'^100|[1-9]?\d$', ds):
        return True
    return False


def read_patient_file(file_name):
    parsed_date = []
    parsed_age = []
    parsed_gender = []
    with open(file_name, 'r', encoding="iso-8859-1") as fin:
        f = fin.read().splitlines()
        for idx, e in enumerate(f):
            if (is_date(e)):
                date = parse(e, dayfirst=True)
                parsed_date.append(date)
            if (is_age(e)):
                parsed_age.append(e)
                parsed_gender.append(f[idx + 1])
    parsed_date.sort()
    age = int(parsed_age[0])
    sex = ['F' if float(parsed_gender[0] == 'נ') else 'M']  # True: Female, False: Male
    recording_date = parsed_date[0]
    analysis_date = parsed_date[1]
    return recording_date, analysis_date, age, sex[0]


def read_timestamp(file_name, start_flag=131, chunk_byte_size=4):
    @dataclass
    class SystemControlData:
        data_tag: int = None
        byte_size: int = None
        continuation_flag: int = None
        start_offset: int = None
        data: bytes = None

    # system_control_data_list = read_control_file(file_name, chunk_byte_size=chunk_byte_size)
    system_control_data_list = list()
    with open(file_name, "rb") as control_file:
        while True:
            data_tag = control_file.read(chunk_byte_size)
            if data_tag:
                chunk = SystemControlData()
                chunk.data_tag = int.from_bytes(data_tag, 'little', signed=True)
                chunk.byte_size = int.from_bytes(control_file.read(chunk_byte_size), 'little', signed=True)
                chunk.continuation_flag = int.from_bytes(control_file.read(chunk_byte_size), 'little', signed=True)
                chunk.start_offset = int.from_bytes(control_file.read(chunk_byte_size), 'little', signed=True)
                chunk.data = control_file.read(chunk.byte_size)
                system_control_data_list.append(chunk)
            else:
                break
    block = next((x for x in system_control_data_list if x.data_tag == start_flag), None)
    vals = np.frombuffer(block.data, dtype=np.int32)
    start_Seconds = int(vals[0] // 128)
    start_time = datetime.timedelta(seconds=start_Seconds)
    end_Seconds = int(vals[1] // 128)
    end_time = datetime.timedelta(seconds=end_Seconds)
    return start_Seconds, end_Seconds


def read_ecg_file(file, b=12, dynamic_range=10, decimal_point=5):
    f = open(file, "rb")
    raw_data = np.fromfile(f, dtype="<u2")

    # remove the 4 MSB which is documented as control data
    # use 0xfff0 if it's the first 4 bits (LSB)
    channel_data = raw_data & 0x0fff

    quantization_step_size = dynamic_range / (2 ** b - 1)

    return np.round(quantization_step_size * channel_data - (dynamic_range / 2), decimal_point)


def read_arrhevnt_file(file):
    f = open(file, "rb")
    raw_data = np.fromfile(f, dtype="<u4")
    array_size = 59000
    num_col = 14
    dummy_vals = [2147483648, 2147483647, 4294967295]
    if len(raw_data) == array_size * num_col:
        raw_data, next_chron_event = raw_data[array_size:], raw_data[:array_size]
        raw_data, previous_chron_event = raw_data[array_size:], raw_data[:array_size]
        raw_data, next_hier_event = raw_data[array_size:], raw_data[:array_size]
        raw_data, previous_hier_event = raw_data[array_size:], raw_data[:array_size]
        raw_data, event_family = raw_data[array_size:], raw_data[:array_size]
        raw_data, event_class = raw_data[array_size:], raw_data[:array_size]
        raw_data, event_start_time = raw_data[array_size:], raw_data[:array_size]
        raw_data, event_end_time = raw_data[array_size:], raw_data[:array_size]
        raw_data, event_length = raw_data[array_size:], raw_data[:array_size]
        raw_data, event_rate = raw_data[array_size:], raw_data[:array_size]
        raw_data, event_rate_position = raw_data[array_size:], raw_data[:array_size]
        raw_data, event_status = raw_data[array_size:], raw_data[:array_size]
        raw_data, element_head_index = raw_data[array_size:], raw_data[:array_size]
        raw_data, element_tail_index = raw_data[array_size:], raw_data[:array_size]
    else:
        print("data size error")

    new_event_family = np.delete(event_family,
                                 np.arange(event_family.shape[0])[np.in1d(event_family[:], dummy_vals)])
    new_previous_chron_event = np.delete(previous_chron_event,
                                         np.arange(previous_chron_event.shape[0])[
                                             np.in1d(previous_chron_event[:], dummy_vals)])
    new_next_chron_event = np.delete(next_chron_event,
                                     np.arange(next_chron_event.shape[0])[np.in1d(next_chron_event[:], dummy_vals)])
    new_next_hier_event = np.delete(next_hier_event,
                                    np.arange(next_hier_event.shape[0])[np.in1d(next_hier_event[:], dummy_vals)])
    new_event_class = np.delete(event_class,
                                np.arange(event_class.shape[0])[np.in1d(event_class[:], dummy_vals)])
    new_event_start_time = np.delete(event_start_time,
                                     np.arange(event_start_time.shape[0])[np.in1d(event_start_time[:], dummy_vals)])
    new_event_end_time = np.delete(event_end_time,
                                   np.arange(event_end_time.shape[0])[np.in1d(event_end_time[:], dummy_vals)])
    new_event_length = np.delete(event_length,
                                 np.arange(event_length.shape[0])[np.in1d(event_length[:], dummy_vals)])
    new_event_rate = np.delete(event_rate,
                               np.arange(event_rate.shape[0])[np.in1d(event_rate[:], dummy_vals)])
    new_event_rate_position = np.delete(event_rate_position,
                                        np.arange(event_rate_position.shape[0])[
                                            np.in1d(event_rate_position[:], dummy_vals)])
    new_event_status = np.delete(event_status,
                                 np.arange(event_status.shape[0])[np.in1d(event_status[:], dummy_vals)])
    new_element_head_index = np.delete(element_head_index,
                                       np.arange(element_head_index.shape[0])[
                                           np.in1d(element_head_index[:], dummy_vals)])
    new_element_tail_index = np.delete(element_tail_index,
                                       np.arange(element_tail_index.shape[0])[
                                           np.in1d(element_tail_index[:], dummy_vals)])

    return pd.DataFrame({'beginning': new_event_start_time, 'end': new_event_end_time, 'length': new_event_status,
                         'class': new_event_class, 'family': new_event_family})


def combtime_file_reader(file):  # , start_recording):
    f = open(file, "rb")
    data = np.fromfile(f, dtype="<u4")
    # df = pd.DataFrame({'beat_time': [start_recording + datetime.timedelta(seconds=x / 128) for x in data]})
    # relative = [start_recording + (x / 128) for x in data]  # with respect to true time start
    # absolute = [(x / 128) for x in data]  # with respect to time 0
    return data  # absolute, relative


def combflag_file_reader(file_name):
    # <u4 is for little endian 32 bit int
    f = open(file_name, "rb")
    raw_data = np.fromfile(f, dtype="<u4")
    # each array will contain True / False based on the flag (bit)
    recorder_clip = raw_data & (0x1 << 31) > 0
    speed_error = raw_data & (0x1 << 30) > 0
    signal_limit = raw_data & (0x1 << 29) > 0
    tape_check = raw_data & (0x1 << 28) > 0
    morphology_print_status_msb = raw_data & (0x1 << 27) > 0
    morphology_print_status_lsb = raw_data & (0x1 << 26) > 0
    paced_machine = raw_data & (0x1 << 25) > 0
    inhibit_machine = raw_data & (0x1 << 24) > 0
    shape = (raw_data & (0xff << 16)) >> 16
    hrv_included_beat = raw_data & (0x1 << 15) > 0
    operator_af_beat = raw_data & (0x1 << 14) > 0
    dropped_beat_raw = raw_data & (0x1 << 13) > 0
    dropped_beat = raw_data & (0x1 << 12) > 0
    pause = raw_data & (0x1 << 11) > 0
    early = raw_data & (0x1 << 10) > 0
    premature_normal_timing = raw_data & (0x1 << 9) > 0
    premature_aberrant_timing = raw_data & (0x1 << 8) > 0
    brady_rate = raw_data & (0x1 << 4) > 0
    svt_rate = raw_data & (0x1 << 3) > 0
    vt_rate = raw_data & (0x1 << 2) > 0

    return shape
