# FSA-Net.Pytorch
**[CVPR19] FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image**
## My Results
BIWI dataset(trained on 300W-LP dataset)：
Intel I7 7500U CPU.
max inference time：58.09 ms ; min inference time：6.44 ms; average inference time：8.81 ms
 -pitch: 5.11 -yaw: 4.53 -roll: 3.89 -mae: 4.51
## Official implemention

https://github.com/shamangary/FSA-Net

## 
It is a very excellent paper as I think. The method may be used in other regression problem, so I implement it in Pytorch for further studying.

### Comparison video
(Baseline **Hopenet:** https://github.com/natanielruiz/deep-head-pose)
<img src="https://github.com/shamangary/FSA-Net/blob/master/Compare_AFLW2000_gt_Hopenet_FSA.gif" height="320"/>

### (New!!!) Fast and robust demo with SSD face detector (2019/08/30)
<img src="https://github.com/shamangary/FSA-Net/blob/master/FSA_SSD_demo.gif" height="300"/>

### Webcam demo

| Signle person (LBP) | Multiple people (MTCNN)|
| --- | --- |
| <img src="https://github.com/shamangary/FSA-Net/blob/master/webcam_demo.gif" height="220"/> | <img src="https://github.com/shamangary/FSA-Net/blob/master/webcam_demo_cvlab_citi.gif" height="220"/> |


| Time sequence | Fine-grained structure|
| --- | --- |
| <img src="https://github.com/shamangary/FSA-Net/blob/master/time_demo.png" height="160"/> | <img src="https://github.com/shamangary/FSA-Net/blob/master/heatmap_demo.png" height="330"/> |



### Results
<img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table1.png" height="220"/><img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table2.png" height="220"/><img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table3.png" height="220"/>


## Paper


### PDF
https://github.com/shamangary/FSA-Net/blob/master/0191.pdf


### Paper authors
**[Tsun-Yi Yang](https://scholar.google.com/citations?user=WhISCE4AAAAJ&hl=en), [Yi-Ting Chen](https://sites.google.com/media.ee.ntu.edu.tw/yitingchen/), [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/index_zh.html), and [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)**


## Abstract
This paper proposes a method for head pose estimation from a single image. Previous methods often predicts head poses through landmark or depth estimation and would require more computation than necessary. Our method is based on regression and feature aggregation. For having a compact model, we employ the soft stagewise regression scheme. Existing feature aggregation methods treat inputs as a bag of features and thus ignore their spatial relationship in a feature map. We propose to learn a fine-grained structure mapping for spatially grouping features before aggregation. The fine-grained structure provides part-based information and pooled values. By ultilizing learnable and non-learnable importance over the spatial location, different variant models as a complementary ensemble can be generated. Experiments show that out method outperforms the state-of-the-art methods including both the landmark-free ones and the ones based on landmark or depth estimation. Based on a single RGB frame as input, our method even outperforms methods utilizing multi-modality information (RGB-D, RGB-Time) on estimating the yaw angle. Furthermore, the memory overhead of the proposed model is 100× smaller than that of previous methods.

