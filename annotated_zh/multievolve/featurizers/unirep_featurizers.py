# -*- coding: utf-8 -*-
# 中文注释版：本文件是 Codex 生成的阅读副本，原始论文代码未被修改。
# 文件作用：UniRep 特征模块，使用 jax-unirep 或 evotuned 参数生成序列表征。
# 阅读方法：先看这些“中文注释”理解结构，再回到原始源码核对实现细节。

from jax_unirep.featurize import get_reps
from jax_unirep.utils import load_params

from multievolve.featurizers.base_featurizers import BaseFeaturizer

UNIREP_MODEL_SIZES = [1900, 256, 64]


# 中文注释：特征化器类 `UnirepBaseFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class UnirepBaseFeaturizer(BaseFeaturizer):
    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(
        self,
        protein=None,
        use_cache=False,
        model_locations=None,
        model_type="unirep",
        model_size=1900,
        **kwargs,
    ):
        super().__init__(model_type, protein, use_cache, **kwargs)
        self.model_locations = model_locations
        self.update_model_name(model_size)
        self.load_params()

    # 中文注释：函数 `update_model_name`：执行本模块中的一个局部处理步骤。
    def update_model_name(self, model_size):
        # Validate model size
        assert model_size in UNIREP_MODEL_SIZES, "Model size must be 1900, 256, or 64."
        self.model_size = model_size
        self.model_type = self.model_type + str(model_size)

    # 中文注释：读取/加载函数 `load_params`：负责从文件、缓存或输入对象中取出后续流程需要的数据。
    def load_params(self):
        self.params = load_params(self.model_locations, self.model_size)[1]

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs, **kwargs):
        h_avg, h_final, c_final = get_reps(
            seqs=seqs, params=self.params, mlstm_size=self.model_size
        )

        return h_avg

# 中文注释：特征化器类 `EvotunedUnirepFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class EvotunedUnirepFeaturizer(UnirepBaseFeaturizer):
    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, model_type="evotuned_unirep", **kwargs):
        super().__init__(model_type=model_type, **kwargs)