import numpy as np
import math


class DataLoader(object):

    def __init__(self, separate_char, max_len, min_len, padding_char):
        self.separate_char = separate_char
        self.max_len = max_len
        self.min_len = min_len
        self.padding_char = padding_char

    # 扫描文件 获得最大序列长度，最小序列长度和最大题号、最小题号
    def scan_file(self, path):
        max_length = 0
        min_length = 1e9
        max_q_id = -1
        # f_data = open(path, 'r')
        with open(path, 'r') as f_data:
            for lineID, line in enumerate(f_data):
                if lineID % 3 == 0:
                    length = int(line.replace(self.separate_char, "").strip())
                    max_length = max(max_length, length)
                    min_length = min(min_length, length)
                elif lineID % 3 == 1:
                    q_ids = self._split_and_check(line)
                    max_q_id = max(max_q_id, max(q_ids))
        return max_length, min_length, max_q_id

    # 对question_id和response进行编码
    @staticmethod
    def encode_input(question_ids, responses, max_q_id):
        assert len(question_ids) == len(responses)
        encoded_vec = []
        for idx, item in enumerate(question_ids):
            new_item = item + responses[idx] * max_q_id
            # new_item += 1
            encoded_vec.append(new_item)
        return encoded_vec

    @staticmethod
    def decode_input(encoded_vec, max_q_id):
        question_ids = []
        responses = []
        for idx, item in enumerate(encoded_vec):
            item -= 1
            target = int(item / max_q_id)
            q_id = item % max_q_id
            question_ids.append(q_id)
            responses.append(target)
        return question_ids, responses

    def padding_list_to_fix_length(self, row_list, length):
        return row_list + [self.padding_char] * (length - len(row_list))

    def padding_target(self, row_list, length):
        return [self.padding_char] * (length - len(row_list)) + row_list

    # data format
    # 15
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
    def prepare_model_data(self, path, max_q_id):
        f_data = open(path, 'r')
        encoded_data = []
        target_index_data = []
        target_response_data = []
        global q_ids  # 存放原始题目id
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 1:
                q_ids = self._split_and_check(line)
            elif lineID % 3 == 2:
                resp = self._split_and_check(line)
                # 丢弃长度小于min_len的序列
                if len(q_ids) >= self.min_len:
                    # 最大长度截断
                    q_ids = q_ids[:self.max_len]
                    resp = resp[:self.max_len]
                    # 对原始长度序列的one-hot编码
                    encoded_vec = self.encode_input(q_ids[:-1], resp[:-1], max_q_id)
                    encoded_response = self.encode_input(q_ids[1:], resp[1:], max_q_id)
                    # 获取原始序列的target index
                    target_index_vec = q_ids[1:]
                    target_response_vec = resp[1:]
                    # padding到相同长度
                    encoded_vec = self.padding_list_to_fix_length(encoded_vec, self.max_len)
                    target_index_vec = self.padding_target(target_index_vec, self.max_len)
                    encoded_response = self.padding_target(encoded_response, self.max_len)
                    encoded_data.append(encoded_vec)
                    target_index_data.append(target_index_vec)
                    target_response_data.append(encoded_response)

        return encoded_data, target_index_data, target_response_data

    # 按照分隔符分割句子 并去掉最后的空值
    def _split_and_check(self, line):
        line = line.replace("\n", "").strip(',')
        res = line.split(self.separate_char)
        if len(res[len(res) - 1]) == 0:
            res = res[:-1]
        result = [int(m) for m in res]
        return result


# data = DataLoader(",", 20, 1, 0)
# max_length, min_length, max_q_id = data.scan_file("./data/oj/1.csv")
# print(max_length, min_length, max_q_id)
# # q = [0, 1, 2, 3, 4]
# # r = [1, 0, 0, 1, 0]
# # print(DATA.encode_input(q, r, 1, 12))
# # print(data.decode_input(data.encode_input(q, r, 1, 12), 1, 12))
# # print(data.get_target_index(q, 0))
#
# print(data.prepare_model_data("./data/oj/1.csv", max_q_id))
# print("Done")