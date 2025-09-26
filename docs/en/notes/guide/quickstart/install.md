---
title: Installation
icon: material-symbols-light:download-rounded
createTime: 2025/06/09 10:29:31
permalink: /en/guide/install/
---
# Installation
This section introduces how to install DataFlow. If you only want to quickly use the Pipelines and operators provided by DataFlow, please refer to the [General User Installation](#General User Installation) to install the stable official version. If you want to contribute code to the DataFlow repository as a developer and submit a Pull Request, please refer to the [Developer Installation](#Developer Installation) to install the dev version in the repository.

## General User Installation

If you do not have a GPU for local inference and only need to use API and CPU functionalities, you can configure the Python environment and install the official version of DataFlow as follows👇

```shell
conda create -n dataflow python=3.10 
conda activate dataflow

pip install open-dataflow
```

If you want to use a local GPU for inference, you need to use the following command:
```shell
conda create -n dataflow python=3.10 
conda activate dataflow

pip install open-dataflow[vllm]
```

> Dataflow supports environments with Python>=3.10.

You can use the following command to check if the installation is correct:
```shell
dataflow -v
```

If the installation is successful and DataFlow is the latest release version, you will see:
```log
open-dataflow codebase version: 0.0.2
        Checking for updates...
        Local version:  0.0.2
        PyPI newest version:  0.0.2
You are using the latest version: 0.0.2.
```
For the exact latest version number, please refer to the [GitHub Release](https://github.com/OpenDCAI/DataFlow/releases) and the [PyPI page](https://pypi.org/project/open-dataflow/).

Additionally, there is a `dataflow env` command to view the current hardware and software environment, which can be used for reporting bugs.

## Developer Installation

DataFlow developers can install using the following commands:

If you do not need local GPU inference:
```shell
conda create -n dataflow python=3.10
conda activate dataflow

git clone https://github.com/OpenDCAI/DataFlow
cd DataFlow
pip install -e .
```

If you need local GPU inference:
```shell
conda create -n dataflow python=3.10
conda activate dataflow

git clone https://github.com/OpenDCAI/DataFlow
cd DataFlow
pip install -e .[vllm]
```

You can use the following command to check if the installation is correct:
```shell
dataflow -v
```

If the installation is successful and DataFlow is the latest release version, you will see:
```log
open-dataflow codebase version: 0.0.2
        Checking for updates...
        Local version:  0.0.2
        PyPI newest version:  0.0.2
You are using the latest version: 0.0.2.
```

Additionally, there is a `dataflow env` command to view the current hardware and software environment, which can be used for reporting bugs.

With this setup, any modifications you make to the DataFlow package locally will be immediately updated in your Python environment, facilitating development. Once development is complete, you can also submit a PR to contribute your new operators and pipelines to the main repository.