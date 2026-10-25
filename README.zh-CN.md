<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <img src="assets/logo-light.svg" alt="CLAUSE" width="520">
</picture>

<p><strong>面向知识图谱、受预算约束的多智能体上下文构建</strong></p>

<p>
  <a href="https://arxiv.org/abs/2509.21035"><img src="https://img.shields.io/badge/ICLR%202026-Conference-1F4E79" alt="ICLR 2026"></a>
  <a href="https://arxiv.org/abs/2509.21035"><img src="https://img.shields.io/badge/arXiv-2509.21035-B31B1B?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
</p>

<p>Yang Zhao<sup>*</sup>, Chengxiao Dai<sup>*</sup>, Wei Zhuo, Yue Xiu, Dusit Niyato<br>
<sub><sup>*</sup>共同第一作者</sub></p>

<p><a href="README.md">English</a> | <strong>简体中文</strong></p>

</div>

## 概述

本仓库是 *CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering*（ICLR 2026）的官方实现。CLAUSE 把多跳问答的上下文构建看作知识图谱上的序贯决策：Subgraph Architect 编辑问题子图，Path Navigator 沿推理路径前进或回溯，Context Curator 选出交给阅读器大模型的证据。三个智能体用 LC-MAPPO（带多头集中式 critic 的拉格朗日约束多智能体 PPO）联合训练，每个问题都有边编辑数、交互步数和选中 token 三种预算。实验在 HotpotQA（distractor）和本地 Wikidata5M 知识图谱上进行。

## 框架

<p align="center">
  <img src="assets/framework.png" alt="CLAUSE framework" width="100%">
</p>

## 安装

需要 Python 3.11 或更高版本。

```bash
git clone https://github.com/Erikdai/clause.git
cd clause
pip install -r requirements.txt
cp .env.example .env    # 填写大模型 API
```

HotpotQA（distractor）和 Wikidata5M 下载到 `data/`：

```bash
mkdir -p data/hotpotqa data/wikidata5m
HF=https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor
curl -L -o data/hotpotqa/validation.parquet  $HF/validation-00000-of-00001.parquet
curl -L -o data/hotpotqa/train-00000.parquet $HF/train-00000-of-00002.parquet
curl -L -o data/wikidata5m/wikidata5m_all_triplet.txt.gz "https://www.dropbox.com/s/563omb11cxaqr83/wikidata5m_all_triplet.txt.gz?dl=1"
curl -L -o data/wikidata5m/wikidata5m_alias.tar.gz "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1"
tar -xzf data/wikidata5m/wikidata5m_alias.tar.gz -C data/wikidata5m
```

## 运行

```bash
python -m clause.prepare
python -m clause.train
python -m clause.evaluate --llm
```

| 脚本 | 内容 |
| --- | --- |
| `prepare.py` | 本地 Wikidata5M 知识图谱、HotpotQA 训练 / 验证 / 测试划分及其嵌入，写入 `data/processed/` |
| `train.py` | 用 LC-MAPPO 训练三个智能体；checkpoint 保存在 `runs/clause/`，`best.pt` 是验证集奖励最高的一次 |
| `evaluate.py` | 在测试集上按预算推理，并计算阅读器回答的 EM 和 F1；`--caps edge,lat,tok` 设定更紧的预算，`--lam edge,lat,tok` 在推理时设定价格 |

`train.py` 的每个设置都可以用 `--<名字>`（训练）或 `--env-<名字>`（环境）修改；`evaluate.py` 支持 `--ckpt` 和 `--split`。结果写入 `results/`，阅读器的回答缓存在 `results/llm_cache/`。

全部设置在 `clause/config.py`。回答由任意兼容 OpenAI 接口的大模型生成，在 `.env` 中设置 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`。

## 目录结构

```
.
├── assets/
│   ├── framework.png       框架图
│   ├── logo-light.svg      Logo（浅色）
│   └── logo-dark.svg       Logo（深色）
├── clause/
│   ├── config.py           路径、预算、环境和训练设置
│   ├── prepare.py          本地知识图谱、问题划分和嵌入
│   ├── data.py             知识图谱、冻结编码器、token 计数和缓存资源
│   ├── env.py              三个智能体的批量环境、预算和证据池
│   ├── models.py           智能体打分网络和多头集中式 critic
│   ├── train.py            LC-MAPPO 训练
│   ├── evaluate.py         按预算推理与回答评测
│   └── reader.py           阅读器大模型和回答指标
├── .env.example            大模型 API 配置模板
└── requirements.txt        固定版本的依赖
```

## 引用

```bibtex
@misc{zhao2026clauseagenticneurosymbolicknowledge,
      title={CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering}, 
      author={Yang Zhao and Chengxiao Dai and Wei Zhuo and Yue Xiu and Dusit Niyato},
      year={2026},
      eprint={2509.21035},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.21035}, 
}
```

## 许可证

[MIT](LICENSE)
