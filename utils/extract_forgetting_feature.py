import numpy as np
import math

class ExtractForget:
    def __init__(self, question_number, base=2):
        '''
        @params
        seq_len: the length of this sequence
        base: the base of log
        question_number: total number of question, 110 for assist2009
        '''
        self.question_number = int(question_number)
        self.base = base

    # 返回每一个序列的repeated_time_sequence_with_log
    def extract_repeated_time_gap_with_log(self, sequence, seq_len):
        max_len = math.floor(np.math.log(seq_len, self.base)) + 1
        sub_rtg = [max_len]
        rtg = [0] * (self.question_number + 1)

        for i in range(len(sequence)):
            if sequence[i] == 0:
                sub_rtg.append(max_len)
                continue

            j = i - 1
            while j >= 0:
                if sequence[j] == sequence[i]:
                    rtg[sequence[i]] = i - j
                    sub_rtg.append(math.floor(np.math.log(rtg[sequence[i]], self.base)))
                    break
                if j == 0:
                    sub_rtg.append(max_len)
                j -= 1
        return sub_rtg


    # 返回一个序列每一个元素的最近重复时间
    # min: 1 max: 201
    def extract_repeated_time_gap(self, sequence, seq_len):
        max_len = seq_len + 1
        sub_rtg = [max_len]
        rtg = [0] * (self.question_number + 1)

        for i in range(len(sequence)):
            if sequence[i] == 0:
                sub_rtg.append(max_len)
                continue

            j = i - 1
            while j >= 0:
                if sequence[j] == sequence[i]:
                    rtg[sequence[i]] = i - j
                    sub_rtg.append(rtg[sequence[i]])
                    break
                if j == 0:
                    sub_rtg.append(max_len)
                j -= 1
        return sub_rtg


    # 返回每一个序列的past_trail_counts_wirh_log
    def extract_past_trail_counts_with_log(self, sequence, seq_len):
        sub_ptc = []
        ptc = [0] * (self.question_number + 1)
        for i in range(len(sequence)):
            sub_ptc.append(math.floor(np.math.log(ptc[sequence[i]], self.base)) if ptc[sequence[i]] != 0 else 0)
            ptc[sequence[i]] += 1
        return sub_ptc

    
    # 返回一个序列每一个元素的过去尝试次数
    # min: 0 max: 190+ (200)
    def extract_past_trail_counts(self, sequence, seq_len):
        sub_ptc = []
        ptc = [0] * (self.question_number + 1)

        for i in range(len(sequence)):
            sub_ptc.append(ptc[sequence[i]]  if ptc[sequence[i]] != 0 else 0)
            ptc[sequence[i]] += 1
        return sub_ptc
    
    '''
    # 返回每个序列后一道题和前一道题的时间差（没有时间戳，都用[1]）
    def extract_sequence_time_gap(self, sequence, seq_len):
        sub_stg = []
        for i in range(len(sequence)):
            sub_stg.append(1)
        return sub_stg
    '''