#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np

def dynamic_chunk_segmentation(Batch_data, m, C, n=1):
    """
    Implementation function of the proposed Dynamic-Chunk-Segmentation as a general data preprocessing step.
    The proposed approach can always map originally different length inputs into fixed size and fix number of data chunks.
    
    Expected I/O:
        Input: batch data list contains different length (time-dim) of 2D feature maps (i.e., 2D numpy matrix)
        Output: fixed 3D dimension numpy array with the shape= (batch-size*C, m, feat-dim)
    
    *** Note *** This function can't process sequence length that is less than the given m!
                 Please make sure all your input data's lengths are always greater then the given m.
    
    Args:
          Batch_data$ (list): list of different length 2D numpy array for batch training data
                   m$ (int) : chunk window length (i.e., number of frames within a chunk),
                              e.g., 1sec=62frames, if LLDs extracted by hop size 16ms then 16ms*62=0.992sec~=1sec
                   C$ (int) : number of data chunks splitted for each sentence
                   n$ (int) : scaling factor to increase number of chunks splitted in a sentence
    """
    num_shifts = n*C-1
    Split_Data = []
    for i in range(len(Batch_data)):
        data = Batch_data[i]
        # checking valid lenght of the input data
        if len(data)<m:
            raise ValueError("input data length is less than the given m, please decrease m!")
        # chunk-shifting size is depending on the length of input data => dynamic step size
        step_size = int(int(len(data)-m)/num_shifts)      
        # calculate index of chunks
        start_idx = [0]
        end_idx = [m]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii+1)*step_size])
            end_idx.extend([end_idx[0] + (iii+1)*step_size])    
        # output split data
        for iii in range(len(start_idx)):
            Split_Data.append( data[start_idx[iii]: end_idx[iii]] )    
    return np.array(Split_Data) # stack as fixed size 3D output
