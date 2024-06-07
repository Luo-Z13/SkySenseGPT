# SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding

## Introduction

In this project, we propose the FIT-RS (Remote Sensing Fine-Grained Instruction Tuning) dataset, which contains 1,800,851 high-quality instruction samples covering various vision-language comprehension tasks. FIT-RS aims to enhance the fine-grained comprehension ability of Remote Sensing Large Multi-Modal Models (RSLMMs), specifically their ability to understand semantic relationships among objects in complex remote sensing scenes.

In addition, we constructed the FIT-RSRC (Remote Sensing Relation Comprehension) Benchmark, which adopts the common-used single-choice format and CircularEval strategy. It includes high-quality distractor options derived from commonsense word lists, as well as unanswerable questions, aiming to evaluate the Remote Sensing Relation Comprehension capabilities of existing LMMs.


## Table of Contents
- [Dataset Description](#dataset-description)
- [Download Links](#download-links)
- [Citation](#citation)
- [License](#license)

### Dataset Description

<ul>
  <li><strong>FIT-RS</strong></li>
      <p align="justify">
      FIT-RS is a large-scale fine-grained instruction tuning dataset, which contains 1,800,851 high quality instruction samples, aiming at enhancing the fine-grained comprehension ability of RSLMMs.
      </p>
    <p align="center">
  <img src="overview.jpg" alt="Introduction" width="90%" />
</p>

  <li>
    <strong>FIT-RSRC</strong><br>
    <p align="justify">
      Given the current lack of a publicly available benchmark for comprehensive and quantitative evaluation of existing LMMs in remote sensing relation understanding, we propose the FIT-RSRC (Remote Sensing Relation Comprehension) benchmark. It is designed in the form of single-choice questions, containing four different types of questions and high-quality distractor options. Following the mainstream general benchmark, FIT-RSRC employs CircularEval as the evaluation strategy.
    </p>
    <p align="center">
  <img src="RSRC.jpg" alt="Introduction" width="90%" />
    </p>
  </li>
</ul>

### Download Links
- **FIT-RS**: A fine-grained instruction tuning dataset.
- **[FIT-RSRC](https://huggingface.co/datasets/ll-13/FIT-RS)**: A benchmark for remote sensing relation comprehension evaluation.
- **SkySenseGPT**: A remote sensing large multi-modal model.

### License
This project is released under the [Apache 2.0 license](LICENSE).

### Citation
- TODO: Add citation information here.
