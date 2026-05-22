# -*- coding: utf-8 -*-
# 中文注释版：本文件是 Codex 生成的阅读副本，原始论文代码未被修改。
# 文件作用：ProtT5 特征模块，使用 HuggingFace T5 模型生成蛋白序列 embedding。
# 阅读方法：先看这些“中文注释”理解结构，再回到原始源码核对实现细节。

import ankh
import numpy as np
import torch

from multievolve.featurizers.base_featurizers import BaseFeaturizer
from transformers import T5Tokenizer, T5EncoderModel
import re

# 中文注释：特征化器类 `ProtT5BaseFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class ProtT5BaseFeaturizer(BaseFeaturizer):
    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, 
                protein=None, 
                use_cache=False,
                model_version=None, 
                batch_size=968,
                model_type="ProtT5",
                **kwargs):
            
        super().__init__(model_type,protein, use_cache, **kwargs)

        self.batch_size = batch_size
        self.model_version = model_version

    # 中文注释：函数 `featurize_prott5`：执行本模块中的一个局部处理步骤。
    def featurize_prott5(self, seqs):

        if self.model_version == 'prot_t5_xl_u50':
            self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(self.device)
        else:
            raise ValueError(f"Invalid model version: {self.model_version}")

        input_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seqs]

        seq_batch = []

        for i in range(0, len(input_seqs), self.batch_size):
            batch = input_seqs[i:i + self.batch_size]
            # tokenize sequences and pad up to the longest sequence in the batch
            ids = self.tokenizer(batch, add_special_tokens=True, padding="longest")

            input_ids = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

            # generate embeddings
            with torch.no_grad():
                embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)

            seq_batch.append(embeddings['last_hidden_state'].mean(axis=1).cpu().numpy())

        return np.concatenate(seq_batch)


# 中文注释：特征化器类 `ProtT5_XL_U50_EmbedFeaturizer`：把蛋白序列、突变或语言模型输出转换为可训练模型使用的数值表示。
class ProtT5_XL_U50_EmbedFeaturizer(ProtT5BaseFeaturizer):
    # 中文注释：构造函数：保存输入参数，初始化对象状态，并准备后续方法需要的属性。
    def __init__(self, 
                protein=None, 
                use_cache=False,
                model_version="prot_t5_xl_u50", 
                batch_size=968,
                model_type="ProtT5_XL_U50_Embed",
                **kwargs):
        super().__init__(protein, use_cache, model_version, batch_size, model_type, **kwargs)

    # 中文注释：子类真正实现特征计算的地方；父类通常负责缓存和批处理，这里负责具体编码。
    def custom_featurizer(self, seqs):

        X = self.featurize_prott5(seqs)
        return X