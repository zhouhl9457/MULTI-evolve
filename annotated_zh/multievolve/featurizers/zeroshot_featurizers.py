# -*- coding: utf-8 -*-
# 中文注释版：本文件是 Codex 生成的阅读副本，原始论文代码未被修改。
# 文件作用：零样本特征模块：把 ESM、MSA、CSCS、ProSE、ESM-IF 的零样本分数转成模型特征。
# 阅读方法：先看这些“中文注释”理解结构，再回到原始源码核对实现细节。

from Bio import SeqIO

from multievolve.featurizers.base_featurizers import BaseFeaturizer
from multievolve.featurizers.model_choices import FEATURE_MODELS
from multievolve.utils.data_utils import find_mutations_multithreaded

# 中文注释：零样本特征化器基类，把突变与语言模型分数连接起来。
class ZeroshotBaseFeaturizer(BaseFeaturizer):
    """Base class for zero-shot featurizers.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotBaseFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta',      # Path to wild-type sequence
        model_locations=[]        # Paths to model files
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        wt_file=None,
        model_type="zeroshot",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model_type, protein, use_cache, **kwargs)
        self.model_locations = model_locations
        self.wt_file = wt_file
        self.wt_seq = str(SeqIO.read(self.wt_file, "fasta").seq)

    # 中文注释：函数 `featurize_zeroshot`：执行本模块中的一个局部处理步骤。
    def featurize_zeroshot(
        self, seqs, model_locations, wt_file, zeroshot_model, **kwargs
    ):
        """
        Featurizes sequences using zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            zeroshot_model (callable): Zero-shot prediction function.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Zero-shot prediction scores.
        """
        assert (self.wt_file is not None) or (
            wt_file is not None
        ), "No wt sequence provided."
        assert (self.model_locations is not None) or (
            model_locations is not None
        ), "No model locations provided."

        wt_file = wt_file or self.wt_file
        model_locations = model_locations or self.model_locations

        wt_seq = str(SeqIO.read(self.wt_file, "fasta").seq)
        model_locations = self.model_locations

        mutations = find_mutations_multithreaded(wt_seq, seqs)

        # make sure to remove model_locations and sequence from kwargs
        kwargs.pop("model_locations", None)
        kwargs.pop("sequence", None)
        kwargs['device'] = self.device
        X = zeroshot_model(
            mutations, model_locations=model_locations, sequence=wt_seq, **kwargs
        )

        X = X.reshape(-1, 1)

        return X


# 中文注释：用 ESM 零样本分数描述突变效果。
class ZeroshotESMFeaturizer(ZeroshotBaseFeaturizer):
    """Class for ESM zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotESMFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta'       # Path to wild-type sequence
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["zeroshot_esm"],
        wt_file=None,
        model_type="zeroshot_esm",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: ESM zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_esm as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            **kwargs
        )
        return X


# 中文注释：用 MSA Transformer 零样本分数描述突变效果。
class ZeroshotMSAFeaturizer(ZeroshotBaseFeaturizer):
    """Class for MSA zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.
        msa_file (str): Path to MSA file.

    Example Usage:

    featurizer = ZeroshotMSAFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta',      # Path to wild-type sequence
        msa_file='msa.fasta'     # Path to MSA file
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["zeroshot_msa"],
        wt_file=None,
        msa_file=None,
        model_type="zeroshot_msa",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            msa_file (str): Path to MSA file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )
        self.msa_file = msa_file

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using MSA zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: MSA zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_msa as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            msa_file=self.msa_file,
            **kwargs
        )
        return X


# 中文注释：特征化器类 `ZeroshotCSCSFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class ZeroshotCSCSFeaturizer(ZeroshotBaseFeaturizer):
    """Class for CSCS zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotCSCSFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta'       # Path to wild-type sequence
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        wt_file=None,
        model_type="zeroshot_cscs",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using CSCS zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: CSCS zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_cscs as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            **kwargs
        )
        return X


# 中文注释：特征化器类 `ZeroshotCSCSGramFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class ZeroshotCSCSGramFeaturizer(ZeroshotBaseFeaturizer):
    """Class for CSCS-Gram zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotCSCSGramFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta'       # Path to wild-type sequence
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        wt_file=None,
        model_type="zeroshot_cscs_gram",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using CSCS-Gram zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: CSCS-Gram zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_cscs_gram as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            **kwargs
        )
        return X


# 中文注释：特征化器类 `ZeroshotCSCSSemFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class ZeroshotCSCSSemFeaturizer(ZeroshotBaseFeaturizer):
    """Class for CSCS-Sem zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotCSCSSemFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta'       # Path to wild-type sequence
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        wt_file=None,
        model_type="zeroshot_cscs_sem",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using CSCS-Sem zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: CSCS-Sem zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_cscs_sem as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            **kwargs
        )
        return X


# 中文注释：特征化器类 `ZeroshotProseFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class ZeroshotProseFeaturizer(ZeroshotBaseFeaturizer):
    """Class for ProSE zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.

    Example Usage:

    featurizer = ZeroshotProseFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta'       # Path to wild-type sequence
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["zeroshot_prose"],
        wt_file=None,
        model_type="zeroshot_prose",
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            **kwargs
        )

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ProSE zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: ProSE zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_prose as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            **kwargs
        )
        return X


# 中文注释：用结构感知的 ESM-IF 零样本分数描述突变效果。
class ZeroshotESMIFFeaturizer(ZeroshotBaseFeaturizer):
    """Class for ESM-IF zero-shot featurization.

    Attributes:
        model_type (str): Type of featurization model to use.
        name (str): Name of the featurizer.
        protein (str): Name of protein being featurized.
        use_cache (bool): Whether to cache featurization results.
        flatten_features (bool): Whether to flatten output features.
        device (torch.device): Device to use for computation.
        model_locations (list): Paths to model files.
        wt_file (str): Path to wild-type sequence file.
        wt_seq (str): Wild-type protein sequence.
        pdb_file (str): Path to PDB structure file.
        chain_id (str): Chain identifier in PDB file.

    Example Usage:

    featurizer = ZeroshotESMIFFeaturizer(
        protein='protein1',       # Name of protein for caching
        use_cache=True,          # Whether to cache results
        wt_file='wt.fasta',      # Path to wild-type sequence
        pdb_file='struct.pdb',   # Path to structure file
        chain_id='A'             # Chain identifier
    )
    features = featurizer.featurize(sequences)
    """

    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=FEATURE_MODELS["zeroshot_esmif"],
        wt_file=None,
        model_type="zeroshot_esmif",
        pdb_file=None,
        chain_id='A',
        **kwargs
    ):
        """
        Args:
            protein (str): Name of protein being featurized.
            use_cache (bool): Whether to cache results.
            model_locations (list): Paths to model files.
            wt_file (str): Path to wild-type sequence file.
            model_type (str): Type of featurization model.
            pdb_file (str): Path to PDB structure file.
            chain_id (str): Chain identifier in PDB file.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            protein=protein,
            use_cache=use_cache,
            model_locations=model_locations,
            wt_file=wt_file,
            model_type=model_type,
            pdb_file=pdb_file,
            chain_id=chain_id,
            **kwargs
        )

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        """
        Featurizes sequences using ESM-IF zero-shot prediction.

        Args:
            seqs (list): List of sequences to featurize.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: ESM-IF zero-shot prediction scores.
        """
        from multievolve.utils.zeroshot_utils import zero_shot_esm_if as zero_shot

        X = self.featurize_zeroshot(
            seqs,
            model_locations=self.model_locations,
            wt_file=self.wt_file,
            zeroshot_model=zero_shot,
            pdb_file=self.pdb_file,
            chain_id=self.chain_id,
            **kwargs
        )
        return X