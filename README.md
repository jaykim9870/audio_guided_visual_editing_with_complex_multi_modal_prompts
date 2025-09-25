# Audio-Guided Visual Editing with Complex Multi-Modal Prompts

[![arXiv](https://img.shields.io/badge/arXiv-2508.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2508.20379)
[![Conference](https://img.shields.io/badge/BMVC-2025-blue)](https://bmvc2025.org)

This repository contains the official implementation of our paper:

> **Audio-Guided Visual Editing with Complex Multi-Modal Prompts**  
> Hyeonyu Kim, Seokhoon Jeong, Seonghee Han, Chanhyuk Choi, and Taehwan Kim  
> *British Machine Vision Conference (BMVC), 2025*

📄 [Read the paper on arXiv](https://arxiv.org/abs/2508.20379)

---

## 🚀 Overview
- Accepted to **BMVC 2025**.  
- Resources will be made publicly available in the coming months. Stay tuned!
- [PIEBench-multi](https://docs.google.com/spreadsheets/d/e/2PACX-1vTAZZ5a02aRxfjTeQmauQ9NvtE1XUeOxI90pdzf5IdDVzM2aQOm8_UtTRHRIdG-Ew/pubhtml)
- [DAVIS-multi](https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3g3Qp-ucEDL9y3DYDlMQFU-fIfGIy1RpRb-QjqlookCaMKpZhJ_LacM0x3R-XOkVigPRiiHW1yXOD/pubhtml)

---

# Instruction

## Installation

1. Clone any-to-any (https://github.com/microsoft/i-Code)

```
#clone CoDi anywhere else
cd ..
git clone https://github.com/microsoft/i-Code

# bring CoDi to our repo
cd anytoedit
mkdir third_party
mv ../i-Code/i-Code-V3 ./third_party/CoDi
```

2. Download CoDi checkpoint and place it to ./checkpoints.

https://github.com/microsoft/i-Code/tree/main/i-Code-V3

3. Download our inverse projection layers and place it to ./checkpoints.

https://drive.google.com/file/d/1of9pBAGmAfdfvJ2KXGlOWnPAs6K5V2fb/view?usp=sharing


## Inference

Modify ./configs/infer.yaml with your desired data and output path. 

```
python infer.py
```

## 📌 Citation
If you find this work useful, please consider citing:

```bibtex
@misc{kim2025audioguidedvisualeditingcomplex,
      title={Audio-Guided Visual Editing with Complex Multi-Modal Prompts}, 
      author={Hyeonyu Kim and Seokhoon Jeong and Seonghee Han and Chanhyuk Choi and Taehwan Kim},
      year={2025},
      eprint={2508.20379},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.20379}, 
}
