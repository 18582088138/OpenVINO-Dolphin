<div align="center">
  <img src="./assets/dolphin.png" width="300">
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2505.14059">
    <img src="https://img.shields.io/badge/Paper-arXiv-red">
  </a>
  <a href="https://huggingface.co/ByteDance/Dolphin-1.5">
    <img src="https://img.shields.io/badge/HuggingFace-Dolphin-yellow">
  </a>
  <a href="https://github.com/bytedance/Dolphin">
    <img src="https://img.shields.io/badge/Code-Github-green">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-lightgray">
  </a>
  <br>
</div>

<br>

<div align="center">
  <img src="./assets/demo.gif" width="800">
</div>

# Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting

Dolphin (**Do**cument Image **P**arsing via **H**eterogeneous Anchor Prompt**in**g) is a novel multimodal document image parsing model (**0.3B**) following an analyze-then-parse paradigm. This repository contains the demo code and pre-trained models for Dolphin.

## 📑 Overview

Document image parsing is challenging due to its complexly intertwined elements such as text paragraphs, figures, formulas, and tables. Dolphin addresses these challenges through a two-stage approach:

1. **🔍 Stage 1**: Comprehensive page-level layout analysis by generating element sequence in natural reading order
2. **🧩 Stage 2**: Efficient parallel parsing of document elements using heterogeneous anchors and task-specific prompts

<div align="center">
  <img src="./assets/framework.png" width="680">
</div>

Dolphin achieves promising performance across diverse page-level and element-level parsing tasks while ensuring superior efficiency through its lightweight architecture and parallel parsing mechanism.

<!-- ## 🚀 Demo
Try our demo on [Demo-Dolphin](https://huggingface.co/spaces/ByteDance/Dolphin). -->

## 📅 Changelog
- 🔥 **2025.11.06** Our new model is coming soon！ 
- 🔥 **2025.10.16** Released *Dolphin-1.5* model. While maintaining the lightweight 0.3B architecture, this version achieves significant parsing improvements. (Dolphin 1.0 moved to [v1.0 branch](https://github.com/bytedance/Dolphin/tree/v1.0))
- 🔥 **2025.07.10** Released the *Fox-Page Benchmark*, a manually refined subset of the original [Fox dataset](https://github.com/ucaslcl/Fox). Download via: [Baidu Yun](https://pan.baidu.com/share/init?surl=t746ULp6iU5bUraVrPlMSw&pwd=fox1) | [Google Drive](https://drive.google.com/file/d/1yZQZqI34QCqvhB4Tmdl3X_XEvYvQyP0q/view?usp=sharing).
- 🔥 **2025.06.30** Added [TensorRT-LLM support](https://github.com/bytedance/Dolphin/blob/master/deployment/tensorrt_llm/ReadMe.md) for accelerated inference！
- 🔥 **2025.06.27** Added [vLLM support](https://github.com/bytedance/Dolphin/blob/master/deployment/vllm/ReadMe.md) for accelerated inference！
- 🔥 **2025.06.13** Added multi-page PDF document parsing capability.
- 🔥 **2025.05.21** Our demo is released at [link](http://115.190.42.15:8888/dolphin/). Check it out!
- 🔥 **2025.05.20** The pretrained model and inference code of Dolphin are released.
- 🔥 **2025.05.16** Our paper has been accepted by ACL 2025. Paper link: [arXiv](https://arxiv.org/abs/2505.14059).

## 📈 Performance

<table style="width:90%; border-collapse: collapse; text-align: center;">
    <caption>Comprehensive evaluation of document parsing on Fox-Page and Dolphin-Page</caption>
    <thead>
        <tr>
            <th style="text-align: center !important;">Model</th>
            <th style="text-align: center !important;">Fox-Page-en<sup>Edit</sup>&#x2193;</th>
            <th style="text-align: center !important;">Fox-Page-zh<sup>Edit</sup>&#x2193;</th>
            <th style="text-align: center !important;">Dolphin-Page-<sup>Edit</sup>&#x2193;</th>
            <th style="text-align: center !important;">Evg<sup>Edit</sup>&#x2193;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Dolphin</td>
            <td>0.0114</td>
            <td>0.0131</td>
            <td>0.1028</td>
            <td>0.0424</td>
        </tr>
        <tr>
            <td>Dolphin-1.5</td>
            <td><strong>0.0074</strong></td>
            <td><strong>0.0077</strong></td>
            <td><strong>0.0743</strong></td>
            <td><strong>0.0298</strong></td>
        </tr>
    </tbody>
</table>

<table style="width:90%; border-collapse: collapse; text-align: center;">
    <caption>Comprehensive evaluation of document parsing on OmniDocBench (v1.5)</caption>
    <thead>
        <tr>
            <th style="text-align: center !important;">Model</th>
            <th style="text-align: center !important;">Overall&#x2191;</th>
            <th style="text-align: center !important;">Text<sup>Edit</sup>&#x2193;</th>
            <th style="text-align: center !important;">Formula<sup>CDM</sup>&#x2191;</th>
            <th style="text-align: center !important;">Table<sup>TEDS</sup>&#x2191;</th>
            <th style="text-align: center !important;">Table<sup>TEDS-S</sup>&#x2191;</th>
            <th style="text-align: center !important;">Read Order<sup>Edit</sup>&#x2193;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Dolphin</td>
            <td>74.67</td>
            <td>0.125</td>
            <td>67.85</td>
            <td>68.70</td>
            <td>77.77</td>
            <td>0.124</td>
        </tr>
        <tr>
            <td>Dolphin-1.5</td>
            <td><strong>83.21</strong></td>
            <td><strong>0.092</strong></td>
            <td><strong>80.78</strong></td>
            <td><strong>78.06</strong></td>
            <td><strong>84.10</strong></td>
            <td><strong>0.080</strong></td>
        </tr>
    </tbody>
</table>

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ByteDance/Dolphin.git
   cd Dolphin
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained models of *Dolphin-1.5*:

   Visit our Huggingface [model card](https://huggingface.co/ByteDance/Dolphin-1.5), or download model by:
   
   ```bash
   # Download the model from Hugging Face Hub
   git lfs install
   git clone https://huggingface.co/ByteDance/Dolphin-1.5 ./hf_model
   # Or use the Hugging Face CLI
   pip install huggingface_hub
   huggingface-cli download ByteDance/Dolphin-1.5 --local-dir ./hf_model
   ```

## ⚡ Inference

Dolphin provides two inference frameworks with support for two parsing granularities:
- **Page-level Parsing**: Parse the entire document page into a structured JSON and Markdown format
- **Element-level Parsing**: Parse individual document elements (text, table, formula)


### 📄 Page-level Parsing

```bash
# Process a single document image
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_1.png 

# Process a single document pdf
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_6.pdf 

# Process all documents in a directory
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs 

# Process with custom batch size for parallel element decoding
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs \
    --max_batch_size 8
```

### 🧩 Element-level Parsing

````bash
# Process element images (specify element_type: table, formula, text, or code)
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path  \
    --element_type [table|formula|text|code]
````

### 🎨 Layout Parsing
````bash
# Process a single document image
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_1.png \
    
# Process a single PDF document
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_6.pdf \

# Process all documents in a directory
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs 
````


## 🌟 Key Features

- 🔄 Two-stage analyze-then-parse approach based on a single VLM
- 📊 Promising performance on document parsing tasks
- 🔍 Natural reading order element sequence generation
- 🧩 Heterogeneous anchor prompting for different document elements
- ⏱️ Efficient parallel parsing mechanism
- 🤗 Support for Hugging Face Transformers for easier integration


## ⚡ONNX Export and inference

### Onnxruntime - Optimum - Export & usage

```bash
# Export onnx fp32
python onnx_export.py --output_dir ./onnx_models/Dolphin-1.5_onnx
# Export onnx fp16
python onnx_export.py --output_dir ./onnx_models/Dolphin-1.5_onnx --fp16

# Inference
python run_onnx_inference \
    --onnx_model_path ./onnx_models/Dolphin-1.5_onnx_fp16 \
    --image_path demo/element_imgs/table.jpg \
    --prompt "Parse the table in the image."

# Demo
python demo_page_onnx.py --onnx_model_path ./onnx_models/Dolphin-1.5_onnx_fp16 --input_path demo/page_imgs/page_6.pdf --save_dir results-dir --max_batch_size 4

# Optional: onnxsim - model optimization

# Install onnxsim
pip install onnxsim

# Run onnxsim for all three onnx models
onnxsim <model/path>.onnx <output/path>.onnx
```


## ⚡OpenVINO Export and inference

### OpenVINO - Optimum-intel - Export & usage

```bash
# Export onnx fp32
python ov_export.py --output_dir ./ov_models/Dolphin-1.5_ov
# Export onnx fp16
python ov_export.py --output_dir ./ov_models/Dolphin-1.5_ov --fp16

# Inference
python ov_run_inference.py \
    --onnx_model_path ./ov_models/Dolphin-1.5_ov_fp16 \
    --image_path ./demo/element_imgs/table.jpg \
    --prompt "Parse the table in the image." \
    --device GPU

# Demo
python ov_demo_page.py --onnx_model_path ./ov_models/Dolphin-1.5_ov_fp16 --input_path demo/page_imgs/page_6.pdf --save_dir results-dir --max_batch_size 4  --device GPU

```



## 📮 Notice
**Call for Bad Cases:** If you have encountered any cases where the model performs poorly, we would greatly appreciate it if you could share them in the issue. We are continuously working to optimize and improve the model.

## 💖 Acknowledgement

We would like to acknowledge the following open-source projects that provided inspiration and reference for this work:
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- [Donut](https://github.com/clovaai/donut/)
- [Nougat](https://github.com/facebookresearch/nougat)
- [GOT](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [MinerU](https://github.com/opendatalab/MinerU/tree/master)
- [Swin](https://github.com/microsoft/Swin-Transformer)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 📝 Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{feng2025dolphin,
  title={Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting},
  author={Feng, Hao and Wei, Shu and Fei, Xiang and Shi, Wei and Han, Yingdong and Liao, Lei and Lu, Jinghui and Wu, Binghong and Liu, Qi and Lin, Chunhui and others},
  journal={arXiv preprint arXiv:2505.14059},
  year={2025}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bytedance/Dolphin&type=Date)](https://www.star-history.com/#bytedance/Dolphin&Date)
