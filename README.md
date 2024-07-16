# AG-NeRF
This is the code repository implementing the paper "AG-NeRF: Attention-guided Neural Radiance Fields for Multi-height Large-scale Outdoor Scene Rendering" published on PRCV 2024
## Datasets
We reuse the training, evaluation datasets from [BungeeNeRF](https://github.com/city-super/BungeeNeRF)

## Training

```
python train.py --config configs/multiscale_google_56Leonard.txt
```

## Evaluation

```
python train.py --config configs/eval_multiscale_google_56Leonard.txt
```

## Cite this work

If you find our work / code implementation useful for your own research, please cite our paper.

```
@article{guo2024ag,
  title={AG-NeRF: Attention-guided Neural Radiance Fields for Multi-height Large-scale Outdoor Scene Rendering},
  author={Guo, Jingfeng and Zhang, Xiaohan and Zhao, Baozhu and Liu, Qi},
  journal={arXiv preprint arXiv:2404.11897},
  year={2024}
}
```

