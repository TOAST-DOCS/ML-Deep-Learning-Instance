## Machine Learning > Deep Learning Instance > 사용 가이드

## Deep Learning Instance 생성

Deep Learning Instance를 사용하려면 먼저 인스턴스를 생성해야 합니다.

![deeplearninginstance_guide_ko_01_20211013.png](https://static.toastoven.net/prod_deep_learning_instance/deeplearninginstance_guide_ko_01_20211013.png)

**Deep Learning Instance 생성** 버튼을 클릭하면 **Machine Learning > Deep Learning Instance > 인스턴스 생성**으로 이동합니다.

Deep Learning Instance에서는 다음과 같은 버전의 소프트웨어가 제공됩니다.

| 소프트웨어 | 버전 | 설치 방식 |
| --- | --- | --- | 
| TensorFlow | 2.4.1 | pip, [참조](https://www.tensorflow.org/install/pip) |
| PyTorch | 1.7.1 | conda, [참조](https://pytorch.org/get-started/previous-versions/) |
| Python | 3.8.11 | conda |
| OS | Ubuntu 18.04 LTS | n/a |
| NVIDIA Driver | 450.102.04 | apt |
| NVIDIA CUDA | 11.0 | apt |
| NVIDIA cuDNN | 8.0.4 | apt |
| NVIDIA NCCL | 2.7.8 | apt |
| NVIDIA TensorRT | 7.1.3 | apt |
| Intel oneAPI MKL | 2021.4.0 | apt |

![deeplearninginstance_guide_ko_02_20211013.png](https://static.toastoven.net/prod_deep_learning_instance/deeplearninginstance_guide_ko_02_20211013.png)

설정을 완료한 후 인스턴스를 생성합니다. 인스턴스 생성에 대한 자세한 내용은 [Instance 개요](http://docs.toast.com/ko/Compute/Instance/ko/overview/)를 참고하시기 바랍니다.

## 설치된 개발 환경 확인

conda 명령어를 사용하여 Miniconda로 설치된 개발 환경을 확인합니다.

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

>[참고]
>
>더 자세한 사용법은 [Miniconda 문서](https://docs.conda.io/en/latest/miniconda.html)를 참고하세요.

## TensorFlow 사용 방법

먼저 TensorFlow 환경을 활성화합니다.

```
(base) root@b64e6a035884:~# conda activate tf2_py38
(tf2_py38) root@b64e6a035884:~#
```

다음과 같이 TensorFlow 훈련을 테스트합니다.

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
# 1개 이상의 GPU를 사용 시 설정
NUM_GPUS=1 # 예) NUM_GPUS=2

python $HOME/models/official/vision/image_classification/mnist_main.py \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --train_epochs=2 \
  --distribution_strategy=mirrored \ # 1개 이상의 GPU를 사용 시 설정
  --num_gpus=$NUM_GPUS \ # 1개 이상의 GPU를 사용 시 설정
  --download

$ chmod +x train.sh
$ ./train.sh
```

>[참고]
>
>더 자세한 사용법은 [TensorFlow 튜토리얼](https://www.tensorflow.org/tutorials)을 참고하세요.

## PyTorch 사용 방법

먼저 PyTorch 환경을 활성화합니다.

```
(tf2_py38) root@b64e6a035884:~# conda deactivate
(base) root@b64e6a035884:~# conda activate pt_py38
(pt_py38) root@b64e6a035884:~#
```

다음과 같이 PyTorch 훈련을 테스트합니다.

```
$ cd ~/
$ git clone https://github.com/pytorch/examples.git
$ cd examples/mnist
$ python main.py --epochs 1
```

>[참고]
>
>더 자세한 사용법은 [PyTorch 튜토리얼](https://pytorch.org/tutorials/)을 참고하세요.
