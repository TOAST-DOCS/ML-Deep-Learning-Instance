## Machine Learning > Deep Learning Instance > 使用ガイド

## Deep Learning Instance作成

Deep Learning Instanceを使用するには、まずインスタンスを作成する必要があります。

![deeplearninginstance_guide_ja_01_20211013.png](https://static.toastoven.net/prod_deep_learning_instance/deeplearninginstance_guide_ja_01_20211013.png)

**Deep Learning Instance作成**ボタンをクリックすると**Machine Learning > Deep Learning Instance > インスタンス作成**に移動します。

Deep Learning Instanceでは次のバージョンのソフトウェアが提供されます。

| Version | Deep Learning Framework | NVIDIA CUDA | NVIDIA cuDNN | OS |
| --- | --- | --- | --- | --- |
| v3.1.0 | TensorFlow 2.12.1 | 11.8 | 8.6 | Ubuntu 22.04 |
| | PyTorch 2.0.1 | 11.8 | 8.7 | Ubuntu 22.04 |

このソフトウェアにはNVIDIA Corporationで<br> 提供したソースコードが含まれています。 [License](https://docs.nvidia.com/deeplearning/cudnn/sla/index.html)

<br>

Deep Learning InstanceはMiniforgeをPythonパッケージマネージャーとして使用し、conda forgeをパッケージリポジトリとして利用しています。
AnacondaやMinicondaをPythonパッケージマネージャとして使用してAnacondaのリポジトリを利用したい場合は、Anacondaのライセンスポリシーを確認してから使用してください。

<br>

![deeplearninginstance_guide_ja_02_20211013.png](https://static.toastoven.net/prod_deep_learning_instance/deeplearninginstance_guide_ja_02_20211013.png)

設定を完了した後にインスタンスを作成します。インスタンス作成の詳細については[Instance概要](http://docs.toast.com/ja/Compute/Instance/ja/overview/)を参照してください。

## インストールされた開発環境の確認

condaコマンドを使用してMinicondaにインストールされた開発環境を確認します。

```
$ conda info --envs
# conda environments:
#
                         /opt/intel/oneapi/intelpython/latest
                         /opt/intel/oneapi/intelpython/latest/envs/2021.4.0
base                  *  /root/miniconda3
pt_py38                  /root/miniconda3/envs/pt_py38
tf2_py38                 /root/miniconda3/envs/tf2_py38
```

>[参考]
>
>詳細な使用方法については[Miniconda文書](https://docs.conda.io/en/latest/miniconda.html)を参考してください。

## TensorFlowの使い方

まずTensorFlow環境を有効にします。

```
(base) root@b64e6a035884:~# conda activate tf2_py38
(tf2_py38) root@b64e6a035884:~#
```

次のようにTensorFlowトレーニングをテストします。

```
$ cd ~/
$ git clone https://github.com/tensorflow/models.git
$ cd models
$ git checkout tags/v2.4.0
$ git status
HEAD detached at v2.4.0
nothing to commit, working tree clean

$ mkdir $HOME/models/model
$ mkdir $HOME/models/dataset
$ vim train.sh
#!/bin/bash


export PYTHONPATH=$HOME/models
export NCCL_DEBUG=INFO
MODEL_DIR=$HOME/models/model
DATA_DIR=$HOME/models/dataset
# 1つ以上のGPUを使用する時に設定
NUM_GPUS=1 # 例) NUM_GPUS=2

python $HOME/models/official/vision/image_classification/mnist_main.py \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --train_epochs=2 \
  --distribution_strategy=mirrored \ # 1つ以上のGPUを使用する時に設定
  --num_gpus=$NUM_GPUS \ # 1つ以上のGPUを使用する時に設定
  --download

$ chmod +x train.sh
$ ./train.sh
```

>[参考]
>
>詳細な使用方法については[TensorFlowチュートリアル](https://www.tensorflow.org/tutorials)を参照してください。

## PyTorchの使い方

まずPyTorch環境を有効にします。

```
(tf2_py38) root@b64e6a035884:~# conda deactivate
(base) root@b64e6a035884:~# conda activate pt_py38
(pt_py38) root@b64e6a035884:~#
```

次のようにPyTorchトレーニングをテストします。

```
$ cd ~/
$ git clone https://github.com/pytorch/examples.git
$ cd examples/mnist
$ python main.py --epochs 1
```

>[参考]
>
>詳細な使用方法については[PyTorchチュートリアル](https://pytorch.org/tutorials/)を参照してください。
