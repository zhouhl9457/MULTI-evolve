# -*- coding: utf-8 -*-
# 中文注释版：本文件是 Codex 生成的阅读副本，原始论文代码未被修改。
# 文件作用：特征缓存工具：创建缓存目录、读取 pickle 缓存、增量更新缓存。
# 阅读方法：先看这些“中文注释”理解结构，再回到原始源码核对实现细节。

# This module contains utility functions for caching and loading feature model data

import numpy as np
import os
import pickle
import sys

# 中文注释：函数 `cache_namespace`：执行本模块中的一个局部处理步骤。
def cache_namespace(fmodel_type, protein):
    """
    Creates a namespace directory for caching feature models of a specific protein.
    
    Args:
    - fmodel_type (str): Type of feature model.
    - protein (str): Name of the protein.
    
    Returns:
    - str: Path to the namespace directory.
    """
    fmodel_type = fmodel_type.replace('/', '-')
    root_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    namespace = f'{root_folder}/proteins/{protein}/feature_cache/{fmodel_type}'
    if not os.path.exists(namespace):
        os.makedirs(namespace)
    return namespace

# 中文注释：读取/加载函数 `load_cache`：负责从文件、缓存或输入对象中取出后续流程需要的数据。
def load_cache(fmodel_type, protein, verbose=1):
    """
    Loads cached feature model data for a specific protein.
    
    Args:
    - fmodel_type (str): Type of feature model.
    - protein (str): Name of the protein.
    - verbose (int): Whether to print the number of sequences loaded from cache.
    
    Returns:
    - dict: Cached data where keys are sequences and values are feature arrays.
    """
    dirname = cache_namespace(fmodel_type, protein)
    
    if not os.path.exists(f'{dirname}/seqs.pkl') or \
       not os.path.exists(f'{dirname}/X.npy'):
        sys.stderr.write(f'Warning: Could not load cache in {dirname}\n')
        return {}

    with open(f'{dirname}/seqs.pkl', 'rb') as f:
        seqs = pickle.load(f)

    X = np.load(f'{dirname}/X.npy')

    cache = {
        seq: X[idx] for idx, seq in enumerate(seqs)
    }
    
    if verbose > 0:
        print(f'Loaded {len(cache)} sequences from cache.')

    return cache

# 中文注释：函数 `update_cache`：执行本模块中的一个局部处理步骤。
def update_cache(fmodel_type, protein, updating_cache_values):
    """
    Update the existing cache with new values.
    
    Args:
    - fmodel_type (str): Type of feature model.
    - protein (str): Name of the protein.
    - updating_cache_values (dict): New values to update the cache with, where keys are sequences and values are feature arrays.
    """
    dirname = cache_namespace(fmodel_type, protein)

    existing_cache = load_cache(fmodel_type, protein, verbose=0)
    new_cache_values = { seq: val for seq, val in updating_cache_values.items() if seq not in existing_cache.keys() }
    # print(f'Existing cache: {len(existing_cache)}')
    print(f'Updating cache with {len(new_cache_values)} new values for {fmodel_type}')
    if len(new_cache_values) > 0:
        updated_cache = existing_cache | new_cache_values
        print(f'Updated cache: {len(updated_cache)}')

        seqs = list(updated_cache.keys())
        X = np.array([ updated_cache[seq] for seq in seqs ])
        
        with open(f'{dirname}/seqs.pkl', 'wb') as f:
            pickle.dump(seqs, f)

        np.save(f'{dirname}/X.npy', X)
