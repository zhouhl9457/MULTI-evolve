# -*- coding: utf-8 -*-
# 中文注释版：本文件是 Codex 生成的阅读副本，原始论文代码未被修改。
# 文件作用：Python 包安装脚本，声明包名、脚本入口、依赖和 Python 版本要求。
# 阅读方法：先看这些“中文注释”理解结构，再回到原始源码核对实现细节。

from setuptools import setup, find_packages

setup(
    name="multievolve",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    scripts=['scripts/p1_train.py',
             'scripts/p2_propose.py',
             'scripts/p3_assembly_design.py',
             'scripts/plm_zeroshot_ensemble.py'],
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.26",
        "pandas>=2.2",
        "matplotlib>=3.10",
        "seaborn>=0.13",
        "scipy>=1.15",
        "biopython>=1.85",
        "scikit-learn>=1.6",
        "scikit-optimize>=0.10",
        "wandb>=0.19",
        "Levenshtein",
        "streamlit>=1.45",
        "fair-esm",
        "biotite>=0.41.2",
    ],
    python_requires=">=3.11",
    authors="Vincent Q. Tran, Matthew Nemeth, and Brian Hie",
    description="MULTI-evolve: model-guided, universal, targeted installation of multi-mutants",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VincentQTran/multievolve",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)