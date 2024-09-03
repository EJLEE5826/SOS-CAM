# A Weakly Supervised Learning Framework Utilizing Enhanced Class Activation Map for Object Detection in Construction Sites
Official implementation of our paper [A Weakly Supervised Learning Framework Utilizing Enhanced Class Activation Map for Object Detection in Construction Sites](https://ieeexplore.ieee.org/document/10584490).
> Jaehun Yang, Eunju Lee, Junehyoung Kwon, Dongmin Lee, Youngbin Kim, Chansik Park, and Doyeop Lee

![architecture](./imgs/SOS-CAM.png)


## Getting Started


### Installation
We test our models under `python=3.8, pytorch=1.12.1, cuda=11.2`.

Clone this repository:
```bash
git clone https://github.com/EJLEE5826/SOS-CAM.git
```
Create a Python virtual environment:
```bash
conda create -n sosnet python=3.9
conda activate sosnet
```

Install other required packages with pip:
```bash
cd sosnet
pip install -r requirements.txt
```

### Training

```bash
python train_cls.py
```


## Citations

```bibtex
@ARTICLE{10584490,
  author={Yang, Jaehun and Lee, Eunju and Kwon, Junehyoung and Lee, Dongmin and Kim, Youngbin and Park, Chansik and Lee, Doyeop},
  journal={IEEE Access}, 
  title={A Weakly Supervised Learning Framework Utilizing Enhanced Class Activation Map for Object Detection in Construction Sites}, 
  year={2024},
  volume={12},
  number={},
  pages={99989-100004},
  doi={10.1109/ACCESS.2024.3423697}}
```

## Acknowledgement

This implementation is built upon [DRS](https://github.com/qjadud1994/DRS). We thank the authors for releasing the codes.

