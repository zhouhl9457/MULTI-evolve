# -*- coding: utf-8 -*-
# 中文注释版：本文件是 Codex 生成的阅读副本，原始论文代码未被修改。
# 文件作用：基础特征化器模块：定义缓存、批量处理和 one-hot/Georgiev/AAIndex 等基础序列编码。
# 阅读方法：先看这些“中文注释”理解结构，再回到原始源码核对实现细节。

from abc import ABC, abstractmethod
import numpy as np
import torch

# root_folder = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(root_folder)

from multievolve.utils.other_utils import AAs
from multievolve.utils.featurizer_utils import seqs_to_georgiev, featurize_aa_idx
from multievolve.utils.cache_utils import load_cache, update_cache


# 中文注释：所有特征化器的抽象基类，统一处理参数、缓存加载、缓存更新和批量 featurize 流程。
class BaseFeaturizer(ABC):
    """Abstract base class for featurizers.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.

    Example Usage:

    featurizer = BaseFeaturizer(
        model_type='onehot',      # Type of featurization used
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        flatten_features=False    # Whether to flatten output features
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, model_type=None, protein=None, use_cache=False, flatten_features=False, **kwargs):
        """
        Args:
            model_type (str): Type of featurization model.
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            flatten_features (bool): Whether to flatten output features.
            **kwargs: Additional keyword arguments.
        """
        self.model_type = model_type
        self.name = str(model_type)
        self.protein = protein
        self.use_cache = use_cache
        self.flatten_features = flatten_features
        self.set_parameters(**kwargs)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    # 中文注释：更新对象参数，通常用于在已有实例上切换配置。
    def set_parameters(self, **kwargs):
        """Sets additional parameters from kwargs."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    # 中文注释：从缓存中读取已经算过的特征，避免重复调用大模型。
    def load_features(self, seqs):
        """
        Loads cached features if they exist.

        Args:
            seqs (list): List of sequences to featurize.

        Returns:
            tuple: (seq_to_feature dict, original sequences, unique sorted sequences)
        """
        print("Loading features...")

        assert (
            self.protein is not None
        ), "No protein specified to cache. Either specify a protein or set use_cache to False."
        original_seqs = seqs
        cache = load_cache(self.model_type, self.protein)
        seq_to_feature = {seq: cache[seq] for seq in seqs if seq in cache}
        seqs = [seq for seq in seqs if seq not in cache]
        print(f"Seqs in cache: {len(seq_to_feature)} | Seqs not in cache: {len(seqs)}")
        del cache  # free up memory

        unique_seqs = {}
        for seq in seqs:
            if seq not in unique_seqs:
                unique_seqs[seq] = len(unique_seqs)

        unique_seqs_sorted = sorted(unique_seqs.keys(), key=lambda k: unique_seqs[k])

        return seq_to_feature, original_seqs, unique_seqs_sorted

    @abstractmethod
    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        Custom featurizer to be implemented in subclasses.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.
        """
        pass

    # 中文注释：把序列或突变列表转换成模型可处理的数值特征。
    def featurize(self, seqs, **kwargs):
        """
        Featurizes input sequences.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array of featurized sequences.
        """
        if self.use_cache:
            seqs_to_feature, original_seqs, unique_seqs_sorted = self.load_features(
                seqs
            )
        else:
            seqs_to_feature = {}
            original_seqs, unique_seqs_sorted = seqs, seqs

        if len(unique_seqs_sorted) > 0:
            X_unique = self.custom_featurizer(unique_seqs_sorted, **kwargs)

            for idx, seq in enumerate(unique_seqs_sorted):
                seqs_to_feature[seq] = X_unique[idx]

        if self.use_cache:
            update_cache(self.model_type, self.protein, seqs_to_feature)

        X = np.array([seqs_to_feature[seq] for seq in original_seqs])

        if self.flatten_features == True:
            X = X.reshape(len(X), -1)

        return X


# 中文注释：把氨基酸序列转换为 one-hot 矩阵，是论文主流程训练神经网络时最基础的输入表示。
class OneHotFeaturizer(BaseFeaturizer):
    """Class for one-hot encoding of sequences.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.

    Example Usage:

    featurizer = OneHotFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        flatten_features=False    # Whether to flatten output features
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, **kwargs) -> None:
        super().__init__(model_type="onehot", **kwargs)

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        One-hot encodes sequences.

        Args:
            seqs (list): List of sequences to encode.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: One-hot encoded sequences.
        """
        data = [[char for char in seq] for seq in seqs]

        from sklearn.preprocessing import OneHotEncoder

        enc = OneHotEncoder(
            categories=([AAs + ["X"]] * len(data[0])),
            sparse_output=False,
        )
        X = enc.fit_transform(data).reshape(len(data), len(data[0]), len(AAs + ["X"]))

        return X


# 中文注释：把氨基酸映射到 Georgiev 理化性质向量。
class GeorgievFeaturizer(BaseFeaturizer):
    """Class for Georgiev encoding of sequences.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.

    Example Usage:

    featurizer = GeorgievFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        flatten_features=False    # Whether to flatten output features
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, **kwargs) -> None:
        super().__init__(model_type="georgiev", **kwargs)

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        Applies Georgiev encoding to sequences.

        Args:
            seqs (list): List of sequences to encode.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Georgiev encoded sequences.
        """
        X = seqs_to_georgiev(seqs)

        return X


# 中文注释：使用 AAIndex/PCA 得到的氨基酸性质向量编码序列。
class AAIdxFeaturizer(BaseFeaturizer):
    """Class for amino acid index encoding of sequences.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.

    Example Usage:

    featurizer = AAIdxFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        flatten_features=False    # Whether to flatten output features
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, **kwargs) -> None:
        super().__init__(model_type="aa_idx", **kwargs)

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        Applies amino acid index encoding to sequences.

        Args:
            seqs (list): List of sequences to encode.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Amino acid index encoded sequences.
        """
        X = featurize_aa_idx(seqs)

        return X