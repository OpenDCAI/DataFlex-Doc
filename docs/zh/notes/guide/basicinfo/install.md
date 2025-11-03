---
title: 安装
icon: material-symbols-light:download-rounded
createTime: 2025/06/09 10:29:31
permalink: /zh/guide/install/
---
# 安装

运行下面的命令来安装：

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory
```

## 使用示例

启动训练的命令与 LlamaFactory 非常相似。以下是一个使用 LESS 的示例，具体原理参考论文 [https://arxiv.org/abs/2402.04333](https://arxiv.org/abs/2402.04333)：

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/less.yaml
```

**注意**：与标准 LlamaFactory 不同，您的 `.yaml` 配置文件中除了需要包含 LlamaFactory 的标准训练参数外，还必须指定 DataFlex 的特定参数。


## 与 LlamaFactory 的集成

DataFlex 完全兼容 LlamaFactory 的配置和使用方式：

1. **配置兼容**：在 LlamaFactory 配置基础上添加 DataFlex 参数
2. **命令一致**：使用 `dataflex-cli` 替代 `llamafactory-cli`
3. **功能保持**：支持所有 LlamaFactory 的原有功能
4. **无缝切换**：可以通过 `train_type: static` 回退到原始训练模式

这种设计确保了用户可以渐进式地采用 DataFlex 的功能，无需对现有工作流进行大幅修改。
