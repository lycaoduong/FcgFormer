
:star: This is the official repo for our paper **Fcg-Former: Identification of Functional Groups in FTIR Spectra Using Enhanced Transformer-Based Model** published in Analytical Chemistry 2024 :star:

<p align="center">
      <a href="https://github.com/lycaoduong/FcgFormer/stargazers">
      <img alt="Stargazers" src="https://img.shields.io/github/stars/lycaoduong/FcgFormer?style=for-the-badge&logo=github&color=f4dbd6&logoColor=D9E0EE&labelColor=302D41"></a>
      <img alt="Python" src="https://img.shields.io/badge/Python%20-%2314354C.svg?style=flat-square&logo=python&logoColor=white" />
      <img alt="Pytorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" />
      <img alt="Hugging Face" src="https://img.shields.io/badge/🤗_Hugging_Face-_oItsMineZ&logoColor=white" />
</p>

# FcgFormer
- Author: lycaoduong
- Email: lycaoduong@gmail.com

## Dataset Download
Contact: lycaoduong@gmail.com

## For training
Run python script train.py with variable parser arguments:
```
--python train.py
```

## From Authors
- [x] Training code
    - [x] Release HuggingFace model
    - [x] Release training source code
- [x] Training environment
    - [x] Release Conda + Pip requirements.txt
    - [ ] Release Dockerfile
- [x] FTIRDataset
  - [x] Release HFSpace: https://huggingface.co/spaces/lycaoduong/FcgFormerApp 
  - [x] Release Dataset: https://huggingface.co/datasets/lycaoduong/FTIR

# **Citation**
```@article{doan2024fcg,
  title={Fcg-Former: Identification of Functional Groups in FTIR Spectra Using Enhanced Transformer-Based Model},
  author={Doan, Vu Hoang Minh and Ly, Cao Duong and Mondal, Sudip and Truong, Thi Thuy and Nguyen, Tan Dung and Choi, Jaeyeop and Lee, Byeongil and Oh, Junghwan},
  journal={Analytical Chemistry},
  year={2024},
  publisher={ACS Publications}
}
