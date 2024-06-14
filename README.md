[![License](https://img.shields.io/badge/license-apache%202.0-60C060.svg)](https://github.com/Luo-Z13/SkySenseGPT?tab=Apache-2.0-1-ov-file)
<img alt="GitHub stars" src="https://img.shields.io/github/stars/Luo-Z13/SkySenseGPT?style=social">

# SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding

## 📢 News and Updates
:fire::fire::fire: Last Updated on 2024.06.13 :fire::fire::fire: 
<!-- - **2024.06.13**: Update readme. -->
- **2024.06.7**: Upload FIT-RSRC dataset to huggingface.


## 📌 Introduction

### [Paper][[Dataset](https://huggingface.co/datasets/ll-13/FIT-RS)][Model][[Code](https://github.com/Luo-Z13/SkySenseGPT)]


In this project, we propose the FIT-RS (Remote Sensing Fine-Grained Instruction Tuning) dataset, which contains 1,800,851 high-quality instruction samples covering various vision-language comprehension tasks. FIT-RS aims to enhance the fine-grained comprehension ability of Remote Sensing Large Multi-Modal Models (RSLMMs), specifically their ability to understand semantic relationships among objects in complex remote sensing scenes.

In addition, we constructed the FIT-RSRC (Remote Sensing Relation Comprehension) Benchmark, which adopts the common-used single-choice format and CircularEval strategy. It includes high-quality distractor options derived from commonsense word lists, as well as unanswerable questions, aiming to evaluate the Remote Sensing Relation Comprehension capabilities of LMMs.



## 🛠️ Table of Contents
- [Dataset and Download](#dataset-and-download)
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
   
- ***FIT-RS:*** A fine-grained remote sensing instruction tuning dataset, containing 1800k instruction samples.
- ***<u>[FIT-RSRC](https://huggingface.co/datasets/ll-13/FIT-RS)</u>:*** A single-choice benchmark for remote sensing relation comprehension evaluation.
- ***SkySenseGPT:*** A remote sensing large multi-modal model.

</ul>



## 🖊️ License
This project is released under the [Apache 2.0 license](LICENSE).


## 🖊️ Citation

If you find this work helpful for your research, please consider giving this repo a star ⭐ and citing our paper:

```bibtex
@article{luo2024sky,
  title={SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding},
  author={Luo, Junwei and Pang, Zhen and Zhang, Yongjun and Wang, Tingzhu and Wang, Linlin and Dang, Bo and Lao, Jiangwei and Wang, Jian and Chen, Jingdong and Tan, Yihua and Li, Yansheng},
  journal={arXiv preprint arXiv:},
  year={2024}
}

@article{li2024scene,
  title={Scene Graph Generation in Large-Size VHR Satellite Imagery: A Large-Scale Dataset and A Context-Aware Approach},
  author={Li, Yansheng and Wang, Linlin and Wang, Tingzhu and Yang, Xue and Wang, Qi and Sun, Xian and Wang, Wenbin and Luo, Junwei and Deng, Youming and Li, Haifeng and Dang, Bo and Zhang, Yongjun and Yan Junchi},
  journal={arXiv preprint arXiv:2406.09410},
  year={2024}
}
```