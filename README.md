# FRCRN: Boosting Feature Representation using Frequency Recurrence for Monaural Speech Enhancement

This repository provides information for **FRCRN** speech enhancement model. It can be retrained and evaluated based on <a href="https://modelscope.cn/models/damo/speech_frcrn_ans_cirm_16k/summary">ModelScope open source platform</a>. The users can either go to the <a href="https://modelscope.cn/models/damo/speech_frcrn_ans_cirm_16k/summary">ModelScope website</a> or follow the steps given below to downloand and install the full pytorch version of FRCRN program. The main model structure was proposed in <a href="https://arxiv.org/abs/2206.07293">FRCRN: Boosting Feature Representation using Frequency Recurrence for Monaural Speech Enhancement</a>.  

## Model Description

FRCRN provides a single-channel noise reduction method which can be used for enhancing speech in various noise environments. Both input and output of FRCRN model are 16kHz time-domain speech waveform signals. The input signal is a noisy audio record from a single microphone, and the output is a noise-suppressed speech audio signal [1]. 

The FRCRN model is developed based on a new framework of **Convolutional Recurrent Encoder-Decoder (CRED)**, which is built on the Convolutional Encoder-Decoder (CED) architecture. CRED can significantly improve the performance of the convolution kernel by improving the limited receptive fields in the convolutions of CED using frequency recurrent layers. In addition, we introduce the Complex Feedforward Sequential Memory Network (CFSMN) to reduce the complexity of the recurrent network, and apply complex-valued network operations to realize the full complex deep model, which not only constructs long sequence speech more effectively, but also can enhance the amplitude and phase of speech at the same time. The model has made good performance in IEEE ICASSP 2022 DNS Challenge.

![model](https://user-images.githubusercontent.com/62317780/203685825-1c349023-c926-45cd-8630-e6289b4d16bd.png)

The input signal of the model is first converted into the complex spectral features through STFT transformation, and then the features are sent to the FRCRN model to predict the complex Ideal Ratio Mask (cIRM) target. The predicted mask is used to multiply the input spectrum to obtain the enhanced spectrum. Finally, the enhanced speech waveform signal is obtained through STFT inverse transformation.

In this implementation, we made a few changes for the model to improve robustness: 1) we use double UNet to further learn masking residual, 2) we simplify CCBAM by using only complex channel attention module, which is equivalent to complex sequeeze and excitation (SE) layer (please refer to <a href=https://arxiv.org/abs/1709.01507>SENet</a> for details). As SE layer aggregates feature information over the whole time indices, it makes the provided model non-causal (In order to ensure the causality during reasoning, cumulative pooling could be used in the time dimension, that is, the pooling operation is gradually performed along the time frame, which is similar to the cumulative layer normalization (cLN) proposed in the <a href=https://arxiv.org/abs/1809.07454>Conv-TasNet</a> model; or a masking matrix could be used like implementing <a href=https://arxiv.org/abs/1706.03762>Transformer</a> attention matrix).

## Installation

After installing <a href="https://github.com/modelscope/modelscope">ModelScope</a>, you can use *speech_frcrn_ans_cirm_16k* for inference. In order to facilitate the usage, the pipeline adds wav file processing logics before and after model processing, which can directly read a wav file and save the output result in the specified wav file.

#### Environment Preparation
This model has been tested under PyTorch v1.8~v1.11 and v1.13. Due to the <a href="https://github.com/pytorch/pytorch/issues/80837">BUG</a> in PyTorch v1.12.0 and v1.12.1, it cannot currently run on v1.12.0 and v1.12.1. If you have already installed this version, please execute the following command to roll back to v1.11:

```
conda install pytorch==1.11 torchaudio torchvision -c pytorch
```

The pipeline of this model uses the third-party library *SoundFile* to process wav files. On the Linux system, users need to manually install the underlying dependency library *libsndfile* of *SoundFile*. On Windows and MacOS, it will be installed automatically without user operation. For detailed information, please refer to the official website of <a href=https://github.com/bastibe/python-soundfile#installation>*SoundFile*</a>. Taking the Ubuntu system as an example, the user needs to execute the following command:

```
sudo apt-get update
sudo apt-get install libsndfile1
```

####  Code Example
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')
result = ans(
    'https://modelscope.cn/api/v1/models/damo/speech_frcrn_ans_cirm_16k/repo?Revision=master&FilePath=examples/speech_with_noise1.wav',
    output_path='output.wav')
```

## Training and Test

#### Dataset
The training data of FRCRN model comes from the DNS-Challenge open source data set, which is provided by the Microsoft team for <a href=https://github.com/microsoft/DNS-Challenge>ICASSP and INTERSPEECH challenges</a>[2]. Our provided model here is trained for the resampled 16k audio. For the convenience of users, we provid a copy of the DNS Challenge 2020 dataset from <a href=https://modelscope.cn/datasets/modelscope/ICASSP_2021_DNS_Challenge/summary>DatasetHub of modelscope</a>, and users can refer to the <a href=https://modelscope.cn/datasets/modelscope/ICASSP_2021_DNS_Challenge/summary>documentation</a> of the dataset in <a href=https://modelscope.cn/datasets/modelscope/ICASSP_2021_DNS_Challenge/summary>DatasetHub of modelscope</a> for the usage.

#### Model Training
The following part is the sample code for model training. We use the test data set as an example in the following example code so that users can quickly verify the environment and collude with the process.

If you want to use more data for training, you can follow the <a href=https://modelscope.cn/datasets/modelscope/ICASSP_2021_DNS_Challenge/summary>dataset documentation</a> to generate training data locally, and then use the code in the comments below to load the dataset instead. Note that the dataset path */your_local_path/ICASSP_2021_DNS_Challenge* should be updated to your local actual path.

```python
import os

from datasets import load_dataset

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

tmp_dir = f'./ckpt'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# Loading dataset
hf_ds = MsDataset.load(
    'ICASSP_2021_DNS_Challenge', split='test').to_hf_dataset()
mapped_ds = hf_ds.map(
    to_segment,
    remove_columns=['duration'],
    batched=True,
    batch_size=36)
mapped_ds = mapped_ds.train_test_split(test_size=150)
# Use below code for real large data training
# hf_ds = load_dataset(
#     '/your_local_path/ICASSP_2021_DNS_Challenge',
#     'train',
#     split='train')
# mapped_ds = hf_ds.map(
#     to_segment,
#     remove_columns=['duration'],
#     num_proc=8,
#     batched=True,
#     batch_size=36)
# mapped_ds = mapped_ds.train_test_split(test_size=3000)
# End of comment

mapped_ds = mapped_ds.shuffle()
dataset = MsDataset.from_hf_dataset(mapped_ds)

kwargs = dict(
    model='damo/speech_frcrn_ans_cirm_16k',
    model_revision='beta',
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    work_dir=tmp_dir)
trainer = build_trainer(
    Trainers.speech_frcrn_ans_cirm_16k, default_args=kwargs)
trainer.train()
```
#### Model Inference
The following code can be used to evaluate and verify the model. We have stored the DNS Challenge 2020 test set on the <a href=https://modelscope.cn/datasets/modelscope/ICASSP_2021_DNS_Challenge/summary>DatasetHub of modelscope</a>, which is convenient for users to download and verify the model.

```
import os
import tempfile

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

hf_ds = MsDataset.load(
    'ICASSP_2021_DNS_Challenge', split='test').to_hf_dataset()
mapped_ds = hf_ds.map(
    to_segment,
    remove_columns=['duration'],
    # num_proc=5, # Comment this line to avoid error in Jupyter notebook
    batched=True,
    batch_size=36)
dataset = MsDataset.from_hf_dataset(mapped_ds)
kwargs = dict(
    model='damo/speech_frcrn_ans_cirm_16k',
    model_revision='beta',
    train_dataset=None,
    eval_dataset=dataset,
    val_iters_per_epoch=125,
    work_dir=tmp_dir)

trainer = build_trainer(
    Trainers.speech_frcrn_ans_cirm_16k, default_args=kwargs)

eval_res = trainer.evaluate()
print(eval_res['avg_sisnr'])
```

#### Evaluation Results
Compared with other SOTA models on the DNS Challenge 2020 official non-blind test set (no reverb), the results are as follows:

![B08384A2-FF54-40F8-900B-84E07D78C785](https://user-images.githubusercontent.com/62317780/203896404-0bcadcbd-8556-4019-95a2-0c25b8748bf3.png)

Measurement Description

- PESQ (Perceptual Evaluation Of Speech Quality) is an objective and full-reference speech quality evaluation method. The score ranges from -0.5 to 4.5. The higher the score, the better the speech quality.
- STOI (Short-Time Objective Intelligibility) reflects the objective evaluation of speech intelligibility by the human auditory perception system. The STOI value is between 0 and 1. The larger the value, the higher the speech intelligibility , the clearer it is.
- SI-SNR (Scale Invariant Signal-to-Noise Ratio) is a scale-invariant signal-to-noise ratio, which reduces the influence of signal changes through regularization on the basis of ordinary signal-to-noise ratios. It is a conventional speech enhancement measurement method for broadband noise distortion.

#### Model Limitation
The FRCRN model provided in ModelScope is a research model trained using the open source data of DNS Challenge. Due to the limited data volume, it may not be able to cover all real data scenarios. If your test data has mismatching conditions with the data used in DNS Challenge due to the recording equipments or recording environments, the output of the model may be not as good as expected. In addition, the performance of the model varies in scenes with multi-speaker interference.

For more details, please refer to the related paper below:

```
[1]
@INPROCEEDINGS{9747578,
  author={Zhao, Shengkui and Ma, Bin and Watcharasupat, Karn N. and Gan, Woon-Seng},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={FRCRN: Boosting Feature Representation Using Frequency Recurrence for Monaural Speech Enhancement}, 
  year={2022},
  pages={9281-9285},
  doi={10.1109/ICASSP43922.2022.9747578}}
```

```
[2]
@INPROCEEDINGS{9747230,
  author={Dubey, Harishchandra and Gopal, Vishak and Cutler, Ross and Aazami, Ashkan and Matusevych, Sergiy and Braun, Sebastian and Eskimez, Sefik Emre and Thakker, Manthan and Yoshioka, Takuya and Gamper, Hannes and Aichner, Robert},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Icassp 2022 Deep Noise Suppression Challenge}, 
  year={2022},
  pages={9271-9275},
  doi={10.1109/ICASSP43922.2022.9747230}}
```
