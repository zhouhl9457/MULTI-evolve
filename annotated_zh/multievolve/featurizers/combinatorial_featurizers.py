# -*- coding: utf-8 -*-
# 中文注释版：本文件是 Codex 生成的阅读副本，原始论文代码未被修改。
# 文件作用：组合特征模块：把 one-hot、ESM、MSA、AAIndex 等多个特征拼接或堆叠给模型使用。
# 阅读方法：先看这些“中文注释”理解结构，再回到原始源码核对实现细节。

from multievolve.featurizers.base_featurizers import *
from multievolve.featurizers.esm_featurizers import *
from multievolve.featurizers.msa_featurizers import *
from multievolve.featurizers.zeroshot_featurizers import *

from multievolve.featurizers.model_choices import FEATURIZE_CHOICES

FEATURIZE_CLASSES = {
    # Dictionary of model names to model classes.
    # Base Featurizers
    "onehot": OneHotFeaturizer,
    "georgiev": GeorgievFeaturizer,
    "aa_idx": AAIdxFeaturizer,
    # MSA Featurizers
    "msa_embed": MSAEmbedFeaturizer,
    "msa_sequence_embed": MSASequenceEmbedFeaturizer,
    "msa_logits": MSALogitsFeaturizer,
    # ESM Featurizers
    "esm_logits": ESMLogitsFeaturizer,
    "esm_embed_1v": ESM1vEmbedFeaturizer,
    "esm_embed_2_3b": ESM2EmbedFeaturizer,
    "esm_embed_2_15b": ESM2_15b_EmbedFeaturizer,
    # Zeroshot Featurizers
    "zeroshot_msa": ZeroshotMSAFeaturizer,
    "zeroshot_esm": ZeroshotESMFeaturizer,
    "zeroshot_prose": ZeroshotProseFeaturizer,
    "zeroshot_cscs": ZeroshotCSCSFeaturizer,
    "zeroshot_cscs_gram": ZeroshotCSCSGramFeaturizer,
    "zeroshot_cscs_sem": ZeroshotCSCSSemFeaturizer,
}


# 中文注释：组合多个基础特征化器，将不同来源的特征合并成一个模型输入。
class CombinatorialFeaturizer():
    """Base class for combining multiple featurizers.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = CombinatorialFeaturizer(
        featurize_methods=['onehot', 'georgiev'],  # List of featurizers to combine
        protein='protein1',                        # Name of protein for caching
        use_cache=True                            # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, featurize_methods, **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        for featurize_method in featurize_methods:
            assert (
                featurize_method in FEATURIZE_CHOICES
            ), f"{featurize_method} not in {FEATURIZE_CHOICES}"

        model_type = "-".join(featurize_methods)
        self.name = str(model_type)

        self.featurizers = {
            featurize_method: FEATURIZE_CLASSES[featurize_method](**kwargs)
            for featurize_method in featurize_methods
        }

    # 中文注释：把序列或突变列表转换成模型可处理的数值特征。
    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using all component featurizers.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined features from all featurizers.
        """
        X = []
        for featurizer in self.featurizers.values():
            X.append(featurizer.featurize(seqs, **kwargs))

        X = np.concatenate(X, axis=-1)

        return X
    
# 中文注释：特征化器类 `ESMAugmentedFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class ESMAugmentedFeaturizer(CombinatorialFeaturizer):
    """Class for combining ESM features with one-hot encoding.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = ESMAugmentedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, featurize_methods=["zeroshot_esm", "onehot"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)
    
    # 中文注释：把序列或突变列表转换成模型可处理的数值特征。
    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM and one-hot encoding.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined ESM and one-hot features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        X.append(featurizer_0.featurize(seqs, **kwargs))
        
        featurizer_1 = list(self.featurizers.values())[1]
        onehot = featurizer_1.featurize(seqs, **kwargs)
        X.append(onehot.reshape(onehot.shape[0], -1))
        
        X = np.concatenate(X, axis=1)

        return X

# 中文注释：特征化器类 `MSAAugmentedFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class MSAAugmentedFeaturizer(CombinatorialFeaturizer):
    """Class for combining MSA features with one-hot encoding.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = MSAAugmentedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, featurize_methods=["zeroshot_msa", "onehot"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)

    # 中文注释：把序列或突变列表转换成模型可处理的数值特征。
    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using MSA and one-hot encoding.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined MSA and one-hot features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        X.append(featurizer_0.featurize(seqs, **kwargs))
        
        featurizer_1 = list(self.featurizers.values())[1]
        onehot = featurizer_1.featurize(seqs, **kwargs)
        X.append(onehot.reshape(onehot.shape[0], -1))
        
        X = np.concatenate(X, axis=1)

        return X

# 中文注释：特征化器类 `OnehotAndGeorgievFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class OnehotAndGeorgievFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot and Georgiev encodings.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndGeorgievFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, featurize_methods=["onehot", "georgiev"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)

# 中文注释：特征化器类 `OnehotAndAAIdxFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class OnehotAndAAIdxFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot and amino acid index encodings.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndAAIdxFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, featurize_methods=["onehot", "aa_idx"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)      

# 中文注释：特征化器类 `OnehotAndESMLogitsFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class OnehotAndESMLogitsFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot encoding with ESM logits.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndESMLogitsFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, featurize_methods=["onehot", "esm_logits"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)
    
    # 中文注释：把序列或突变列表转换成模型可处理的数值特征。
    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using one-hot encoding and ESM logits.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined one-hot and ESM logits features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        x = featurizer_0.featurize(seqs, **kwargs)
        zero_vectors = np.zeros((x.shape[0], 1, x.shape[2]))
        X.append(np.concatenate((zero_vectors, x, zero_vectors), axis=1))
        
        featurizer_1 = list(self.featurizers.values())[1]
        X.append(featurizer_1.featurize(seqs, **kwargs))
        
        X = np.concatenate(X, axis=-1)

        return X

# 中文注释：特征化器类 `OnehotAndESMMSALogitsFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class OnehotAndESMMSALogitsFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot encoding with ESM-MSA logits.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndESMMSALogitsFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, featurize_methods=["onehot", "msa_logits"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)
    
    # 中文注释：把序列或突变列表转换成模型可处理的数值特征。
    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using one-hot encoding and ESM-MSA logits.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined one-hot and ESM-MSA logits features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        x = featurizer_0.featurize(seqs, **kwargs)
        zero_vectors = np.zeros((x.shape[0], 1, x.shape[2]))
        X.append(np.concatenate((zero_vectors, x), axis=1))
        
        featurizer_1 = list(self.featurizers.values())[1]
        X.append(featurizer_1.featurize(seqs, **kwargs))
        
        X = np.concatenate(X, axis=-1)

        return X
    

# 中文注释：特征化器类 `OnehotAndESM2EmbedFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class OnehotAndESM2EmbedFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot encoding with ESM2 embeddings.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndESM2EmbedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, featurize_methods=["onehot", "esm_embed_2_3b"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)

    # 中文注释：把序列或突变列表转换成模型可处理的数值特征。
    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using one-hot encoding and ESM2 embeddings.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined one-hot and ESM2 embedding features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        onehot = featurizer_0.featurize(seqs, **kwargs)
        X.append(onehot.reshape(onehot.shape[0], -1))
        
        featurizer_1 = list(self.featurizers.values())[1]
        X.append(featurizer_1.featurize(seqs, **kwargs))
        
        X = np.concatenate(X, axis=1)

        return X

# 中文注释：特征化器类 `OnehotAndESM2_15bEmbedFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class OnehotAndESM2_15bEmbedFeaturizer(CombinatorialFeaturizer):
    """Class for combining one-hot encoding with ESM2 embeddings.

    Attributes:
        name (str): Name of the combined featurizer.
        featurizers (dict): Dictionary mapping featurizer names to instances.

    Example Usage:
    
    featurizer = OnehotAndESM2EmbedFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True           # Whether to cache results
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, featurize_methods=["onehot", "esm_embed_2_15b"], **kwargs):
        """
        Args:
            featurize_methods (list): List of featurizer names to combine.
            **kwargs: Additional arguments passed to each featurizer.
        """
        super().__init__(featurize_methods, **kwargs)

    # 中文注释：把序列或突变列表转换成模型可处理的数值特征。
    def featurize(self, seqs, **kwargs):
        """
        Featurizes sequences using one-hot encoding and ESM2 embeddings.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional arguments passed to each featurizer.

        Returns:
            np.ndarray: Combined one-hot and ESM2 embedding features.
        """
        X = []
        
        featurizer_0 = list(self.featurizers.values())[0]
        onehot = featurizer_0.featurize(seqs, **kwargs)
        X.append(onehot.reshape(onehot.shape[0], -1))
        
        featurizer_1 = list(self.featurizers.values())[1]
        X.append(featurizer_1.featurize(seqs, **kwargs))
        
        X = np.concatenate(X, axis=1)

        return X