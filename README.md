# Quantised-Self-Attentive-Deep-Neural-Network
Official code for the paper titled "**Verifiable and Energy Efficient Medical Image Analysis with Quantised Self-attentive Deep Neural Networks**"

Paper: [Springer](https://link.springer.com/chapter/10.1007/978-3-031-18523-6_17) [Arxiv](https://arxiv.org/pdf/2209.15287.pdf)

## Abstract:
Convolutional Neural Networks have played a significant role in various medical imaging tasks like classification and segmentation. They provide state-of-the-art performance compared to classical image processing algorithms. However, the major downside of these methods is the high computational complexity, reliance on high-performance hardware like GPUs and the inherent black-box nature of the model. In this paper, we propose quantised stand-alone self-attention based models as an alternative to traditional CNNs. In the proposed class of networks, convolutional layers are replaced with stand-alone self-attention layers, and the network parameters are quantised after training. We experimentally validate the performance of our method on classification and segmentation tasks. We observe 50–80% reduction in model size, 60–80% lesser number of parameters, 40–85% fewer FLOPs and 65–80% more energy efficiency during inference on CPUs.

## Keywords

Stand alone self attention, Quantisation, Medical image analysis, Classification, Segmentation, Compute complexity.


## To-Do
1. Add requirements.txt
2. Add sample images from dataset for unit testing.
3. Remove absolute paths in config and code
4. Re-run notebooks
