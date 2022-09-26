# Dynamic-Chunk-Segmentation
Implementation of the chunking process proposed in [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9442335) as data segmentation preprocessing step.

![Visualization of the chunking process](/images/visualize.png)


# Numeric dummy example
```python
import numpy as np
from chunking_process import dynamic_chunk_segmentation

# generate dummy batch data with some common feature parameters (e.g., wav2vec, LLDs, opensmile...etc)
batch_size = 64
feat_dim = 768          # e.g., wav2vec, some Transformer-based pretrained deep features
frame_hop_size = 0.02   # e.g., wav2vec uses 20ms as feature hop-size in the training setting
time_range = np.random.uniform(low=3, high=15, size=batch_size) # data duration are bounded in 3-15 secs
batch_data_to_train = []
for i in range(batch_size):
    # 2D feat-map with shape= (num_frames, feat_dim)
    feat = np.random.rand(int(time_range[i]/frame_hop_size), feat_dim)
    batch_data_to_train.append(feat)


# pre-defined chunking parameters
m = 50    # chunk-window-size=1sec, since 50frames*20ms=1sec
C = 15    # Tmax=15secs, C=Tmax/1sec=15, where 1sec is the defined chunk-window-size in second unit
n = 1     # use the default setting
batch_chunk_data_to_train = dynamic_chunk_segmentation(Batch_data=batch_data_to_train, m=m, C=C, n=n)
print(batch_chunk_data_to_train.shape)
print("Batch-Size: "+str(batch_size)+"*"+str(C))
print("Chunk-Window-Size (Time): "+str(m))
print("Feat-Dim: "+str(feat_dim))
```


# Usage in PyTorch DataLoader: collate_fn
Simply add the segmentation function as the collation function in torch.DataLoader object.
```python
def collate_fn(batch):
    m = 50
    C = 15
    n = 1
    data, label = zip(*batch)
    batch_chunk_data_to_train = dynamic_chunk_segmentation(Batch_data=data, m=m, C=C, n=n)
    batch_chunk_label_to_train = np.repeat(np.array(label), C)
    return torch.from_numpy(batch_chunk_data_to_train), torch.from_numpy(batch_chunk_label_to_train)



# creating common Torch data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
train_loader = torch.utils.data.DataLoader(training_dataset, 
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           num_workers=12,
                                           pin_memory=True,
                                           collate_fn=collate_fn)
```



# Reference
If you use this code, please cite the following paper:

Wei-Cheng Lin and Carlos Busso, "Chunk-Level Speech Emotion Recognition: A General Framework of Sequence-to-One Dynamic Temporal Modeling"

```
@article{Lin_202x,
    author={W.-C. Lin and C. Busso},
    title={Chunk-Level Speech Emotion Recognition: A General Framework of Sequence-to-One Dynamic Temporal Modeling},
    journal={IEEE Transactions on Affective Computing},
    number={},
    volume={To Appear},
    pages={},
    year={2021},
    month={},
    doi={10.1109/TAFFC.2021.3083821},
}
```
