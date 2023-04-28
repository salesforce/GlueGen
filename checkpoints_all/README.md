### Download Checkpoints
Download the official checkpoints of SD v1 to `./checkpoints_all/checkpoint_sd_v1` as `./checkpoints_all/checkpoint_sd_v1/v1-5-pruned-emaonly.ckpt` (downloaded from https://huggingface.co/runwayml/stable-diffusion-v1-5).

Then follow the README.md (`./stable-diffusion/audioclip/README.md`) of audioclip to download checkpoints  to `./checkpoints_all/audioclip_checkpoint` as  `./checkpoints_all/audioclip_checkpoint/AudioCLIP-Full-Training.pt`.

```
mkdir ./audioclip_checkpoint
cd ./audioclip_checkpoint
wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt
```

Then download the pretrained gluenet checkpoints and save them to `./checkpoints_all/gluenet_checkpoint`:
```
bash ../download_gluenet_checkpoints.sh
```
