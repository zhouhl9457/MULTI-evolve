# -*- coding: utf-8 -*-
# 中文注释版：本文件是 Codex 生成的阅读副本，原始论文代码未被修改。
# 文件作用：特征名称和模型位置映射表，决定某个 featurizer 默认调用哪些预训练模型。
# 阅读方法：先看这些“中文注释”理解结构，再回到原始源码核对实现细节。

from multievolve.featurizers.model_locations import (
    msa_models,
    esm_models,
    prose_models,
    prose_models_cas13,
)

FEATURIZE_CHOICES = [
    # All the implemented featurizers
    # Base Featurizers
    'onehot',
    'georgiev',
    'aa_idx',

    # MSA Featurizers
    'msa_embed',
    'msa_sequence_embed',
    'msa_logits',
    'msa_augmented',

    # ESM Featurizers
    'esm_logits',
    'esm_embed_1v',
    'esm_embed_2_3b',
    'esm_embed_2_15b',
    'esm_augmented',

    # Zeroshot Featurizers
    'zeroshot_msa',
    'zeroshot_esm',
    'zeroshot_cscs',
    'zeroshot_cscs_gram',
    'zeroshot_cscs_sem',
    'zeroshot_prose',
    'zeroshot_esmif',
]

FEATURE_MODELS = {
    # Dictionary of model names to model locations.
    # Base Featurizers
    'onehot': [ None ],
    'georgiev': [ None ],
    'aa_idx': [ None ],

    # MSA Featurizers
    'msa_embed': msa_models,
    'msa_sequence_embed': msa_models,
    'msa_logits': msa_models,
    'msa_augmented': msa_models,

    # ESM Featurizers
    'esm_logits': esm_models[:1],
    'esm_embed_1v': esm_models[:1],
    'esm_embed_2_3b': esm_models[5:6],
    'esm_embed_2_15b': esm_models[7:8],
    'esm_augmented': esm_models[:5],

    # Zeroshot Featurizers
    'zeroshot_msa': msa_models,
    'zeroshot_esm': esm_models[:6],
    'zeroshot_cscs': [ None ],
    'zeroshot_cscs_gram': [ None ],
    'zeroshot_cscs_sem': [ None ],
    'zeroshot_prose': prose_models_cas13,
    'zeroshot_esmif': esm_models[6:7],

    # Prose Featurizers
    'prose_embedmean': prose_models,
    'prose_augmented': prose_models_cas13,
}