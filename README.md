[![License](https://img.shields.io/badge/license-apache%202.0-60C060.svg)](https://github.com/Luo-Z13/SkySenseGPT?tab=Apache-2.0-1-ov-file)
[![Paper](https://img.shields.io/badge/Arxiv-2406.10100-blue)](https://arxiv.org/abs/2406.10100)
[![Paper](https://img.shields.io/badge/Dataset-FITRS-orange)](https://huggingface.co/datasets/ll-13/FIT-RS)
<img alt="GitHub stars" src="https://img.shields.io/github/stars/Luo-Z13/SkySenseGPT?style=social">

# SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding

## 📢 News and Updates
:fire::fire::fire: **Last Updated: August 2025. This project is under active development.**:fire::fire::fire: 

- **[2025.08.08]** Our **SkySense Family** [project page](https://zqcrafts.github.io/SkySense-O/project.html) is now live!
- **[2024.07.22]** The FIT-RSFG benchmark have been uploaded [here](https://huggingface.co/datasets/ll-13/FIT-RS/tree/main/FIT-RSFG) and the evaluation scripts have been released [here](Eval_scripts)! See [Evaluation](#evaluation) for details on how to evaluate.
% - **[2024.07.20]** The FIT-RS dataset (training set 1415k) **categorized by tasks** have been uploaded [here](https://huggingface.co/datasets/ll-13/FIT-RS/blob/main/FIT-RS_Instruction/train_data_of_each_individual_task.zip).
- **[2024.06.17]** Our paper is available in [arxiv](https://arxiv.org/abs/2406.10100)!


## 📌 Introduction

### [[Paper](https://arxiv.org/abs/2406.10100)][[Dataset](https://huggingface.co/datasets/ll-13/FIT-RS)][[Model](https://huggingface.co/ll-13/SkySenseGPT-7B-clip-lora)][[Code](https://github.com/Luo-Z13/SkySenseGPT)]


In this project, we propose the FIT-RS (Remote Sensing Fine-Grained Instruction Tuning) dataset, which contains 1,800,851 high-quality instruction samples covering various vision-language comprehension tasks. FIT-RS aims to enhance the fine-grained comprehension ability of Remote Sensing Large Multi-Modal Models (RSLMMs), specifically their ability to understand semantic relationships among objects in complex remote sensing scenes. Based on FIT-RS, we establish the FIT-RSFG (Remote Sensing Fine-Grained Comprehension) Benchmark to evaluate RSLMMs' ability in fine-grained understanding.

In addition, we constructed the FIT-RSRC (Remote Sensing Relation Comprehension) Benchmark, which adopts the common-used single-choice format and CircularEval strategy. It includes high-quality distractor options derived from commonsense word lists, as well as unanswerable questions, aiming to evaluate the Remote Sensing Relation Comprehension capabilities of LMMs.



## 🛠️ Table of Contents
- [Dataset and Download](#dataset-and-download)
- [Evaluation](#evaluation)
- [License](#license)
- [Citation](#citation)


## ⭐️ Dataset and Download

<ul>
  <li><strong>FIT-RS</strong></li>
      <p align="justify">
      FIT-RS is a large-scale fine-grained instruction tuning dataset, which contains 1,800,851 high quality instruction samples, aiming at enhancing the fine-grained comprehension ability of RSLMMs.
      </p>
    <p align="center">
  <img src="overview.png" alt="Introduction" width="100%" />
</p>

  <li>
    <strong>FIT-RSRC</strong><br>
    <p align="justify">
      Given the current lack of a publicly available benchmark for comprehensive and quantitative evaluation of existing LMMs in remote sensing relation understanding, we propose the FIT-RSRC (Remote Sensing Relation Comprehension) benchmark. It is designed in the form of <strong>single-choice</strong> questions, containing four different types of questions and high-quality distractor options. Following the mainstream general benchmark, FIT-RSRC employs CircularEval as the evaluation strategy.
    </p>
    <p align="center">
  <img src="RSRC.jpg" alt="Introduction" width="100%" />
    </p>
  </li>

   <li><strong>Download Links</strong></li>
   
- ***<u>[FIT-RS](https://huggingface.co/datasets/ll-13/FIT-RS)</u>:*** A fine-grained remote sensing instruction tuning dataset, containing 1800k instruction samples, 1415k for training.
- ***<u>[FIT-RSFG](https://huggingface.co/datasets/ll-13/FIT-RS/tree/main/FIT-RSFG)</u>:*** A fine-grained benchmark for remote sensing vision-language evaluation.
- ***<u>[FIT-RSRC](https://huggingface.co/datasets/ll-13/FIT-RS/tree/main/FIT-RSRC)</u>:*** A single-choice benchmark for remote sensing relation comprehension evaluation.
- ***<u>[SkySenseGPT](https://huggingface.co/ll-13/SkySenseGPT-7B-clip-lora)</u>:*** A remote sensing large multi-modal model, capable of handling complex comprehension tasks like image-level scene graph generation.

</ul>

## Evaluation
1. Download [FIT-RSFG](https://huggingface.co/datasets/ll-13/FIT-RS/tree/main/FIT-RSFG) and [FIT-RSRC](https://huggingface.co/datasets/ll-13/FIT-RS/tree/main/FIT-RSRC) Benchmarks.
2. Install necessary packages as in the [requirements.txt](requirements.txt).
3. See [evaluation.sh](Eval_scripts/evaluation.sh) for evaluation.

## License
This project is released under the [Apache 2.0 license](LICENSE).


## Citation

Our FIT-RS dataset is built based on the [STAR](https://linlin-dev.github.io/project/STAR.html) dataset. If you find this work helpful for your research, please consider giving this repo a star ⭐ and citing our paper:

```bibtex
@article{luo2024sky,
    title={SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding},
    author={Luo, Junwei and Pang, Zhen and Zhang, Yongjun and Wang, Tingzhu and Wang, Linlin and Dang, Bo and Lao, Jiangwei and Wang, Jian and Chen, Jingdong and Tan, Yihua and Li, Yansheng},
    journal={arXiv preprint arXiv:2406.10100},
    year={2024}
}

@article{li2024scene,
    title={STAR: A First-Ever Dataset and A Large-Scale Benchmark for Scene Graph Generation in Large-Size Satellite Imagery},
    author={Li, Yansheng and Wang, Linlin and Wang, Tingzhu and Yang, Xue and Luo, Junwei and Wang, Qi and Deng, Youming and Wang, Wenbin and Sun, Xian and Li, Haifeng and Dang, Bo and Zhang, Yongjun and Yu, Yi and Yan Junchi},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2024},
    publisher={IEEE}}

@inproceedings{zhu2025skysenseo,
  title={Skysense-o: Towards open-world remote sensing interpretation with vision-centric visual-language modeling},
  author={Zhu, Qi and Lao, Jiangwei and Ji, Deyi and Luo, Junwei and Wu, Kang and Zhang, Yingying and Ru, Lixiang and Wang, Jian and Chen, Jingdong and Yang, Ming and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={14733--14744},
  year={2025}
}

@article{wu2025semantic,
  author       = {Wu, Kang and Zhang, Yingying and Ru, Lixiang and Dang, Bo and Lao, Jiangwei and Yu, Lei and Luo, Junwei and Zhu, Zifan and Sun, Yue and Zhang, Jiahao and Zhu, Qi and Wang, Jian and Yang, Ming and Chen, Jingdong and Zhang, Yongjun and Li, Yansheng},
  title        = {A semantic‑enhanced multi‑modal remote sensing foundation model for Earth observation},
  journal      = {Nature Machine Intelligence},
  year         = {2025},
  doi          = {10.1038/s42256-025-01078-8},
  url          = {https://doi.org/10.1038/s42256-025-01078-8}
}

@inproceedings{guo2024skysense,
    author    = {Guo, Xin and Lao, Jiangwei and Dang, Bo and Zhang, Yingying and Yu, Lei and Ru, Lixiang and Zhong, Liheng and Huang, Ziyuan and Wu, Kang and Hu, Dingxiang and He, Huimei and Wang, Jian and Chen, Jingdong and Yang, Ming and Zhang, Yongjun and Li, Yansheng},
    title     = {SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {27672-27683}
}
```

We are thankful to [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) and [GeoChat](https://github.com/mbzuai-oryx/GeoChat) for releasing their models and code as open-source contributions.


