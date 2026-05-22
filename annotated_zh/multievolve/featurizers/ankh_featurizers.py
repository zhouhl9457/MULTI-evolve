# -*- coding: utf-8 -*-
# 中文注释版：本文件是 Codex 生成的阅读副本，原始论文代码未被修改。
# 文件作用：Ankh 蛋白语言模型特征模块，支持 base 和 large 模型 embedding。
# 阅读方法：先看这些“中文注释”理解结构，再回到原始源码核对实现细节。

import ankh
import numpy as np
import torch

from multievolve.featurizers.base_featurizers import BaseFeaturizer

# alternate name: AnkhBaseFeaturizer
# 中文注释：特征化器类 `AnkhFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class AnkhFeaturizer(BaseFeaturizer):
    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, 
                protein=None, 
                use_cache=False,
                model_version=None, 
                batch_size=968,
                model_type="ankh",
                **kwargs):
            
        super().__init__(model_type,protein, use_cache, **kwargs)

        self.batch_size = batch_size
        self.model_version = model_version

    # 中文注释：函数 `featurize_ankh`：执行本模块中的一个局部处理步骤。
    def featurize_ankh(self, seqs):

        if self.model_version == 'large':
            self.model, self.tokenizer = ankh.load_large_model()
        elif self.model_version == 'base':
            self.model, self.tokenizer = ankh.load_base_model()
        else:
            raise ValueError(f"Invalid model version: {self.model_version}")
        self.model.eval()
        self.model.to(self.device)

        input_seqs = [list(seq) for seq in seqs]

        seq_batch = []

        for i in range(0, len(input_seqs), self.batch_size):
            batch = input_seqs[i:i + self.batch_size]
            outputs = self.tokenizer(
                batch, 
                add_special_tokens=True, 
                padding=True, 
                is_split_into_words=True, 
                return_tensors="pt",
            )
            outputs = {key: val.to(self.device) for key, val in outputs.items()}
            with torch.no_grad():
                embeddings = self.model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
            seq_batch.append(embeddings['last_hidden_state'].mean(axis=1).cpu().numpy())

        return np.concatenate(seq_batch)

# alternate name: AnkhBaseEmbedFeaturizer
# 中文注释：特征化器类 `AnkhBaseFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class AnkhBaseFeaturizer(AnkhFeaturizer):
    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, 
                protein=None, 
                use_cache=False,
                model_version="base", 
                batch_size=968,
                model_type="ankh_base",
                **kwargs):
        super().__init__(protein, use_cache, model_version, batch_size, model_type, **kwargs)

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs):

        X = self.featurize_ankh(seqs)
        return X

# alternate name: AnkhLargeEmbedFeaturizer
# 中文注释：特征化器类 `AnkhLargeFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class AnkhLargeFeaturizer(AnkhFeaturizer):
    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, 
                protein=None, 
                use_cache=False,
                model_version="large", 
                batch_size=968,
                model_type="ankh_large",
                **kwargs):
        super().__init__(protein, use_cache, model_version, batch_size, model_type, **kwargs)

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs):
        X = self.featurize_ankh(seqs)
        return X