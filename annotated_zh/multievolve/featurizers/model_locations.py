# -*- coding: utf-8 -*-
# 中文注释版：本文件是 Codex 生成的阅读副本，原始论文代码未被修改。
# 文件作用：预训练模型名称/路径常量表，供 ESM、MSA、ProSE 等模块引用。
# 阅读方法：先看这些“中文注释”理解结构，再回到原始源码核对实现细节。

esm_models = [
    'esm1v_t33_650M_UR90S_1',
    'esm1v_t33_650M_UR90S_2',
    'esm1v_t33_650M_UR90S_3',
    'esm1v_t33_650M_UR90S_4',
    'esm1v_t33_650M_UR90S_5',
    'esm2_t36_3B_UR50D',
    'esm_if1_gvp4_t16_142M_UR50',
    'esm2_t48_15B_UR50D',
]

msa_models = [
    'esm_msa1b_t12_100M_UR50S',
]

prose_models = [
    'data/prose_pretrained_models/prose_dlm_3x1024.sav',
]

prose_models_cas13 = [
    'target/cas13/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_0/_iter0500000.sav',
    'target/cas13/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_1/_iter0400000.sav',
    'target/cas13/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_3/_iter0600000.sav',
    'target/cas13/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_4/_iter0600000.sav',
    'target/cas13/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_5/_iter0600000.sav',
]

prose_models_cas13_old1 = [
    'target/cas13/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_0/_iter0500000.sav',
    'target/cas13/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_1/_iter0400000.sav',
    'target/cas13/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_2/_iter0300000.sav',
]

prose_models_cas13_old = [
    'target/cas13_old/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_1/_iter0300000.sav',
    'target/cas13_old/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_2/_iter0200000.sav',
    'target/cas13_old/prose_rd512_nl3_dr0_ns2000000_si100000_le1280_mr0.1_bs100_wd0_lr0.0001_cs0.98_3/_iter0300000.sav',
]
