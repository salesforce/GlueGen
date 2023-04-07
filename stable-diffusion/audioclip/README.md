# AudioCLIP
## Extending [CLIP](https://github.com/openai/CLIP) to Image, Text and Audio

![Overview of AudioCLIP](images/AudioCLIP-Structure.png)

This repository contains implementation of the models described in the paper [arXiv:2106.13043](https://arxiv.org/abs/2106.13043).

### Downloading Pre-Trained Weights

The pre-trained model can be downloaded from the [releases](https://github.com/AndreyGuzhov/AudioCLIP/releases).

    # AudioCLIP trained on AudioSet (text-, image- and audio-head simultaneously)
    wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt

#### Important Note
If you use AudioCLIP as a part of GAN-based image generation, please consider downloading the [partially](https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Partial-Training.pt) trained model, as its audio embeddings are compatible with the vanilla [CLIP](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt) (based on ResNet-50).

### How to Run the Model

The required Python version is >= 3.7.

#### AudioCLIP

##### On the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset
    python main.py --config protocols/audioclip-esc50.json --Dataset.args.root /path/to/ESC50

##### On the [UrbanSound8K](https://urbansounddataset.weebly.com/) dataset
    python main.py --config protocols/audioclip-us8k.json --Dataset.args.root /path/to/UrbanSound8K
