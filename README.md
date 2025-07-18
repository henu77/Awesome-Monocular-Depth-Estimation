# Awesome Monocular Depth Estimation

A curated list of monocular depth estimation papers.

The list focuses primarily on papers published after 2022, including some particularly outstanding work from earlier years.

精选单目深度估计论文列表。精选并整理了 `2022` 年后发表的单目深度估计论文，同时涵盖部分早期的优秀成果。

## 2025

### [Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](https://arxiv.org/pdf/2501.02464) ![Static Badge](https://img.shields.io/badge/CVPR-FF0000)
[Code](https://github.com/yuliangguo/depth_any_camera) | [Project](https://yuliangguo.github.io/depth-any-camera/) 
<details closed>
<summary>Abstract</summary>
While recent depth foundation models exhibit strong zero-shot generalization, achieving accurate metric depth across diverse camera types-particularly those with large fields of view (FoV) such as fisheye and 360-degree cameras-remains a significant challenge. This paper presents Depth Any Camera (DAC), a powerful zero-shot metric depth estimation framework that extends a perspective-trained model to effectively handle cameras with varying FoVs. The framework is designed to ensure that all existing 3D data can be leveraged, regardless of the specific camera types used in new applications. Remarkably, DAC is trained exclusively on perspective images but generalizes seamlessly to fisheye and 360-degree cameras without the need for specialized training data. DAC employs Equi-Rectangular Projection (ERP) as a unified image representation, enabling consistent processing of images with diverse FoVs. Its core components include pitch-aware Image-to-ERP conversion with efficient online augmentation to simulate distorted ERP patches from undistorted inputs, FoV alignment operations to enable effective training across a wide range of FoVs, and multi-resolution data augmentation to further address resolution disparities between training and testing. DAC achieves state-of-the-art zero-shot metric depth estimation, improving \delta_1 accuracy by up to 50% on multiple fisheye and 360-degree datasets compared to prior metric depth foundation models, demonstrating robust generalization across camera types.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@InProceedings{guo2025depth,
  title={Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera},
  author={Guo, Yuliang and Garg, Sparsh and Miangoleh, S Mahdi H and Huang, Xinyu and Ren, Liu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
</details>

### [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/pdf/2410.02073) ![Static Badge](https://img.shields.io/badge/ICLR-FF0000)
[Code](https://github.com/apple/ml-depth-pro) 
<details closed>
<summary>Abstract</summary>
We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image. Extensive experiments analyze specific design choices and demonstrate that Depth Pro outperforms prior work along multiple dimensions.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@inproceedings{Bochkovskii2024:arxiv,
  author     = {Aleksei Bochkovskii and Ama"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun},
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  booktitle  = {International Conference on Learning Representations},
  year       = {2025}
}
```
</details>

### [Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction](https://arxiv.org/pdf/2409.18124) ![Static Badge](https://img.shields.io/badge/ICLR-FF0000)
[Code](https://github.com/EnVision-Research/Lotus) | [Project](https://lotus3d.github.io/) | [Demo 1](https://huggingface.co/spaces/haodongli/Lotus_Depth) | [Demo 2](https://huggingface.co/spaces/haodongli/Lotus_Normal) 
<details closed>
<summary>Abstract</summary>
Leveraging the visual priors of pre-trained text-to-image diffusion models offers a promising solution to enhance zero-shot generalization in dense prediction tasks. However, existing methods often uncritically use the original diffusion formulation, which may not be optimal due to the fundamental differences between dense prediction and image generation. In this paper, we provide a systemic analysis of the diffusion formulation for the dense prediction, focusing on both quality and efficiency. And we find that the original parameterization type for image generation, which learns to predict noise, is harmful for dense prediction; the multi-step noising/denoising diffusion process is also unnecessary and challenging to optimize. Based on these insights, we introduce Lotus, a diffusion-based visual foundation model with a simple yet effective adaptation protocol for dense prediction. Specifically, Lotus is trained to directly predict annotations instead of noise, thereby avoiding harmful variance. We also reformulate the diffusion process into a single-step procedure, simplifying optimization and significantly boosting inference speed. Additionally, we introduce a novel tuning strategy called detail preserver, which achieves more accurate and fine-grained predictions. Without scaling up the training data or model capacity, Lotus achieves SoTA performance in zero-shot depth and normal estimation across various datasets. It also enhances efficiency, being significantly faster than most existing diffusion-based methods. Lotus' superior quality and efficiency also enable a wide range of practical applications, such as joint estimation, single/multi-view 3D reconstruction, etc.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@inproceedings{li2024lotus,
  title={Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction},
  author={He, Jing and Li, Haodong and Yin, Wei and Liang, Yixun and Li, Leheng and Zhou, Kaiqiang and Liu, Hongbo and Liu, Bingbing and Chen, Ying-Cong},
  booktitle={International Conference on Learning Representations},
  year={2025},

}
```
</details>

### [Scalable Autoregressive Monocular Depth Estimation](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Scalable_Autoregressive_Monocular_Depth_Estimation_CVPR_2025_paper.pdf) ![Static Badge](https://img.shields.io/badge/CVPR-FF0000)
[Code](https://github.com/wjh892521292/DAR) | [Project](https://depth-ar.github.io/) | [Supplementary](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Wang_Scalable_Autoregressive_Monocular_CVPR_2025_supplemental.pdf) 
<details closed>
<summary>Abstract</summary>
This paper proposes a new autoregressive model as an effective and scalable monocular depth estimator. Our idea is simple: We tackle the monocular depth estimation (MDE) task with an autoregressive prediction paradigm, based on two core designs. First, our depth autoregressive model (DAR) treats the depth map of different resolutions as a set of tokens, and conducts the low-to-high resolution autoregressive objective with a patch-wise casual mask. Second, our DAR recursively discretizes the entire depth range into more compact intervals, and attains the coarse-to-fine granularity autoregressive objective in an ordinal-regression manner. By coupling these two autoregressive objectives, our DAR establishes new state-of-the-art (SOTA) on KITTI and NYU Depth v2 by clear margins. Further, our scalable approach allows us to scale the model up to 2.0B and achieve the best RMSE of 1.799 on the KITTI dataset (5% improvement) compared to 1.896 by the current SOTA (Depth Anything). DAR further showcases zero-shot generalization ability on unseen datasets. These results suggest that DAR yields superior performance with an autoregressive prediction paradigm, providing a promising approach to equip modern autoregressive large models (e.g., GPT-4o) with depth estimation capabilities. Project page: https://depth-ar.github.io/
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@InProceedings{Wang_2025_CVPR,
    author    = {Wang, Jinhong and Liu, Jian and Tang, Dongqi and Wang, Weiqiang and Li, Wentong and Chen, Danny and Chen, Jintai and Wu, Jian},
    title     = {Scalable Autoregressive Monocular Depth Estimation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6262-6272}
}
```
</details>

### [SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation](https://openaccess.thecvf.com/content/CVPR2025/papers/Pham_SharpDepth_Sharpening_Metric_Depth_Predictions_Using_Diffusion_Distillation_CVPR_2025_paper.pdf) ![Static Badge](https://img.shields.io/badge/CVPR-FF0000)
[Project](https://sharpdepth.github.io/) | [Supplementary](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Pham_SharpDepth_Sharpening_Metric_CVPR_2025_supplemental.pdf) 
<details closed>
<summary>Abstract</summary>
We propose SharpDepth, a novel approach to monocular metric depth estimation that combines the metric accuracy of discriminative depth estimation methods (e.g., Metric3D, UniDepth) with the fine-grained boundary sharpness typically achieved by generative methods (e.g., Marigold, Lotus). Traditional discriminative models trained on real-world data with sparse ground-truth depth can accurately predict metric depth but often produce over-smoothed or low-detail depth maps. Generative models, in contrast, are trained on synthetic data with dense ground truth, generating depth maps with sharp boundaries yet only providing relative depth with low accuracy. Our approach bridges these limitations by integrating metric accuracy with detailed boundary preservation, resulting in depth predictions that are both metrically precise and visually sharp. Our extensive zero-shot evaluations on standard depth estimation benchmarks confirm SharpDepth effectiveness, showing its ability to achieve both high depth accuracy and detailed representation, making it well-suited for applications requiring high-quality depth perception across diverse, real-world environments.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@InProceedings{Pham_2025_CVPR,
    author    = {Pham, Duc-Hai and Do, Tung and Nguyen, Phong and Hua, Binh-Son and Nguyen, Khoi and Nguyen, Rang},
    title     = {SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {17060-17069}
}
```
</details>

## 2024

### [ECoDepth: Effective Conditioning of Diffusion Models for Monocular Depth Estimation](https://openaccess.thecvf.com/content/CVPR2024/papers/Patni_ECoDepth_Effective_Conditioning_of_Diffusion_Models_for_Monocular_Depth_Estimation_CVPR_2024_paper.pdf) ![Static Badge](https://img.shields.io/badge/CVPR-FF0000)
[Code](https://github.com/aradhye2002/ecodepth) | [Project](https://ecodepth-iitd.github.io/) | [Supplementary](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Patni_ECoDepth_Effective_Conditioning_CVPR_2024_supplemental.pdf) 
<details closed>
<summary>Abstract</summary>
In the absence of parallax cues, a learning based single image depth estimation (SIDE) model relies heavily on shading and contextual cues in the image. While this simplicity is attractive, it is necessary to train such models on large and varied datasets, which are difficult to capture. It has been shown that using embeddings from pretrained foundational models, such as CLIP, improves zero shot transfer in several applications. Taking inspiration from this, in our paper we explore the use of global image priors generated from a pre-trained ViT model to provide more detailed contextual information. We argue that the embedding vector from a ViT model, pre-trained on a large dataset, captures greater relevant information for SIDE than the usual route of generating pseudo image captions, followed by CLIP based text embeddings. Based on the idea, we propose a new SIDE model using a diffusion backbone conditioned on ViT embeddings. Our proposed design establishes a new state-of-the-art (SOTA) for SIDE on NYU Depth v2 dataset, achieving Abs Rel error of 0.059(14% improvement) compared to 0.069 by the current SOTA (VPD). And on KITTI dataset, achieving SqRel error of 0.139 (2% improvement) compared to 0.142 by the current SOTA (GEDepth). For zero shot transfer with a model trained on NYU Depth v2, we report mean relative improvement of (20%, 23%, 81%, 25%) over NeWCRF on (Sun-RGBD, iBims1, DIODE, HyperSim) datasets, compared to (16%, 18%, 45%, 9%) by ZoEDepth.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@InProceedings{Patni_2024_CVPR,
    author    = {Patni, Suraj and Agarwal, Aradhye and Arora, Chetan},
    title     = {ECoDepth: Effective Conditioning of Diffusion Models for Monocular Depth Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {28285-28295}
}
```
</details>

### [IEBins: Iterative Elastic Bins for Monocular Depth Estimation](https://proceedings.neurips.cc/paper_files/paper/2023/file/a61023ce36d21010f1423304f8ec49af-Paper-Conference.pdf) ![Static Badge](https://img.shields.io/badge/NeurIPS-FF0000)
[Code](https://github.com/ShuweiShao/IEBins) | [Supplementary](https://proceedings.neurips.cc/paper_files/paper/2023/file/a61023ce36d21010f1423304f8ec49af-Supplemental-Conference.pdf) 
<details closed>
<summary>Abstract</summary>
Monocular depth estimation (MDE) is a fundamental topic of geometric computer vision and a core technique for many downstream applications. Recently, several methods reframe the MDE as a classification-regression problem where a linear combination of probabilistic distribution and bin centers is used to predict depth. In this paper, we propose a novel concept of iterative elastic bins (IEBins) for the classification-regression-based MDE. The proposed IEBins aims to search for high-quality depth by progressively optimizing the search range, which involves multiple stages and each stage performs a finer-grained depth search in the target bin on top of its previous stage. To alleviate the possible error accumulation during the iterative process, we utilize a novel elastic target bin to replace the original target bin, the width of which is adjusted elastically based on the depth uncertainty. Furthermore, we develop a dedicated framework composed of a feature extractor and an iterative optimizer that has powerful temporal context modeling capabilities benefiting from the GRU-based architecture. Extensive experiments on the KITTI, NYU-Depth-v2 and SUN RGB-D datasets demonstrate that the proposed method surpasses prior state-of-the-art competitors. The source code is publicly available at https://github.com/ShuweiShao/IEBins.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@inproceedings{NEURIPS2023_a61023ce,
  author = {Shao, Shuwei and Pei, Zhongcai and Wu, Xingming and Liu, Zhong and Chen, Weihai and Li, Zhengguo},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
  pages = {53025--53037},
  publisher = {Curran Associates, Inc.},
  title = {IEBins: Iterative Elastic Bins for Monocular Depth Estimation},
 volume = {36},
  year = {2023}
}
```
</details>

### [PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PatchFusion_An_End-to-End_Tile-Based_Framework_for_High-Resolution_Monocular_Metric_Depth_CVPR_2024_paper.pdf) ![Static Badge](https://img.shields.io/badge/CVPR-FF0000)
[Code](https://zhyever.github.io/patchfusion/) 
<details closed>
<summary>Abstract</summary>
Single image depth estimation is a foundational task in computer vision and generative modeling. However prevailing depth estimation models grapple with accommodating the increasing resolutions commonplace in today's consumer cameras and devices. Existing high-resolution strategies show promise but they often face limitations ranging from error propagation to the loss of high-frequency details. We present PatchFusion a novel tile-based framework with three key components to improve the current state of the art: (1) A patch-wise fusion network that fuses a globally-consistent coarse prediction with finer inconsistent tiled predictions via high-level feature guidance (2) A Global-to-Local (G2L) module that adds vital context to the fusion network discarding the need for patch selection heuristics and (3) A Consistency-Aware Training (CAT) and Inference (CAI) approach emphasizing patch overlap consistency and thereby eradicating the necessity for post-processing. Experiments on UnrealStereo4K MVS-Synth and Middleburry 2014 demonstrate that our framework can generate high-resolution depth maps with intricate details. PatchFusion is independent of the base model for depth estimation. Notably our framework built on top of SOTA ZoeDepth brings improvements for a total of 17.3% and 29.4% in terms of the root mean squared error (RMSE) on UnrealStereo4K and MVS-Synth respectively.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@InProceedings{Li_2024_CVPR,
    author    = {Li, Zhenyu and Bhat, Shariq Farooq and Wonka, Peter},
    title     = {PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {10016-10025}
}
```
</details>

### [Scale-Invariant Monocular Depth Estimation via SSI Depth](https://arxiv.org/pdf/2406.09374) ![Static Badge](https://img.shields.io/badge/ACM_SIGGRAPH-FF0000)
[Project](https://yaksoy.github.io/sidepth/) | [Supplementary](https://yaksoy.github.io/papers/SIG24-SI-Depth-Supp.pdf) 
<details closed>
<summary>Abstract</summary>
Existing methods for scale-invariant monocular depth estimation (SI MDE) often struggle due to the complexity of the task, and limited and non-diverse datasets, hindering generalizability in real-world scenarios. This is while shift-and-scale-invariant (SSI) depth estimation, simplifying the task and enabling training with abundant stereo datasets achieves high performance. We present a novel approach that leverages SSI inputs to enhance SI depth estimation, streamlining the network’s role and facilitating in-the-wild generalization for SI depth estimation while only using a synthetic dataset for training. Emphasizing the generation of high-resolution details, we introduce a novel sparse ordinal loss that substantially improves detail generation in SSI MDE, addressing critical limitations in existing approaches. Through in-the-wild qualitative examples and zero-shot evaluation we substantiate the practical utility of our approach in computational photography applications, showcasing its ability to generate highly detailed SI depth maps and achieve generalization in diverse scenarios.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@INPROCEEDINGS{miangolehSIDepth,
author={S. Mahdi H. Miangoleh and Mahesh Reddy and Ya\u{g}{\i}z Aksoy},
title={Scale-Invariant Monocular Depth Estimation via SSI Depth},
booktitle={ACM SIGGRAPH},
year={2024},
}
```
</details>

### [WorDepth: Variational Language Prior for Monocular Depth Estimation](https://openaccess.thecvf.com/content/CVPR2024/papers/Zeng_WorDepth_Variational_Language_Prior_for_Monocular_Depth_Estimation_CVPR_2024_paper.pdf) ![Static Badge](https://img.shields.io/badge/CVPR-FF0000)
[Code](https://github.com/Adonis-galaxy/WorDepth) | [Supplementary](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zeng_WorDepth_Variational_Language_CVPR_2024_supplemental.pdf) 
<details closed>
<summary>Abstract</summary>
Three-dimensional (3D) reconstruction from a single image is an ill-posed problem with inherent ambiguities i.e. scale. Predicting a 3D scene from text description(s) is similarly ill-posed i.e. spatial arrangements of objects described. We investigate the question of whether two inherently ambiguous modalities can be used in conjunction to produce metric-scaled reconstructions. To test this we focus on monocular depth estimation the problem of predicting a dense depth map from a single image but with an additional text caption describing the scene. To this end we begin by encoding the text caption as a mean and standard deviation; using a variational framework we learn the distribution of the plausible metric reconstructions of 3D scenes corresponding to the text captions as a prior. To 'select' a specific reconstruction or depth map we encode the given image through a conditional sampler that samples from the latent space of the variational text encoder which is then decoded to the output depth map. Our approach is trained alternatingly between the text and image branches: in one optimization step we predict the mean and standard deviation from the text description and sample from a standard Gaussian and in the other we sample using a (image) conditional sampler. Once trained we directly predict depth from the encoded text using the conditional sampler. We demonstrate our approach on indoor (NYUv2) and outdoor (KITTI) scenarios where we show that language can consistently improve performance in both. Code: https://github.com/Adonis-galaxy/WorDepth.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@InProceedings{Zeng_2024_CVPR,
    author    = {Zeng, Ziyao and Wang, Daniel and Yang, Fengyu and Park, Hyoungseob and Soatto, Stefano and Lao, Dong and Wong, Alex},
    title     = {WorDepth: Variational Language Prior for Monocular Depth Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {9708-9719}
}
```
</details>

## 2022

### [BinsFormer: Revisiting Adaptive Bins for Monocular Depth Estimation](https://arxiv.org/pdf/2204.00987) ![Static Badge](https://img.shields.io/badge/IEEE_TIP-FF0000)
[Code](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/binsformer.) 
<details closed>
<summary>Abstract</summary>
Monocular depth estimation (MDE) is a fundamental task in computer vision and has drawn increasing attention. Recently, some methods reformulate it as a classification-regression task to boost the model performance, where continuous depth is estimated via a linear combination of predicted probability distributions and discrete bins. In this paper, we present a novel framework called BinsFormer, tailored for the classification-regression-based depth estimation. It mainly focuses on two crucial components in the specific task: 1) proper generation of adaptive bins; and 2) sufficient interaction between probability distribution and bins predictions. To specify, we employ a Transformer decoder to generate bins, novelly viewing it as a direct set-to-set prediction problem. We further integrate a multi-scale decoder structure to achieve a comprehensive understanding of spatial geometry information and estimate depth maps in a coarse-to-fine manner. Moreover, an extra scene understanding query is proposed to improve the estimation accuracy, which turns out that models can implicitly learn useful information from the auxiliary environment classification task. Extensive experiments on the KITTI, NYU, and SUN RGB-D datasets demonstrate that BinsFormer surpasses state-of-the-art MDE methods with prominent margins. Code and pretrained models are made publicly available at https://github.com/zhyever/ Monocular-Depth-Estimation-Toolbox/tree/main/configs/ binsformer.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@ARTICLE{10570231,
  author={Li, Zhenyu and Wang, Xuyang and Liu, Xianming and Jiang, Junjun},
  journal={IEEE Transactions on Image Processing}, 
  title={BinsFormer: Revisiting Adaptive Bins for Monocular Depth Estimation}, 
  year={2024},
  volume={33},
  number={},
  pages={3964-3976},
  keywords={Estimation;Transformers;Task analysis;Decoding;Probabilistic logic;Training;Computer vision;Monocular depth estimation;adaptive bins;multi-scale refinement;auxiliary task;transformer},
  doi={10.1109/TIP.2024.3416065}
}
```
</details>

## 2021

### [AdaBins: Depth Estimation Using Adaptive Bins](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_AdaBins_Depth_Estimation_Using_Adaptive_Bins_CVPR_2021_paper.pdf) ![Static Badge](https://img.shields.io/badge/CVPR-FF0000)
[Code](https://github.com/shariqfarooq123/AdaBins) | [Demo](https://camo.githubusercontent.com/96889048f8a9014fdeba2a891f97150c6aac6e723f5190236b10215a97ed41f3/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667) | [Supplementary](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Bhat_AdaBins_Depth_Estimation_CVPR_2021_supplemental.zip) 
<details closed>
<summary>Abstract</summary>
We address the problem of estimating a high quality dense depth map from a single RGB input image. We start out with a baseline encoder-decoder convolutional neural network architecture and pose the question of how the global processing of information can help improve overall depth estimation. To this end, we propose a transformer-based architecture block that divides the depth range into bins whose center value is estimated adaptively per image. The final depth values are estimated as linear combinations of the bin centers. We call our new building block AdaBins. Our results show a decisive improvement over the state-of-the-art on several popular depth datasets across all metrics. We also validate the effectiveness of the proposed block with an ablation study and provide the code and corresponding pre-trained weights of the new state-of-the-art model.
</details>

<details closed>
<summary>Citation</summary>

```bibtex
@InProceedings{Bhat_2021_CVPR,
    author    = {Bhat, Shariq Farooq and Alhashim, Ibraheem and Wonka, Peter},
    title     = {AdaBins: Depth Estimation Using Adaptive Bins},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {4009-4018}
}
```
</details>
