**Impurities Detection in Edible Bird’s Nest using U-net & AE models**

This project develops custom U-Net & AE models to detect impurities in edible bird’s nest images.

**Cite these articles**

1. Ying-Heng Yeo and Kin-Sam Yen, “Impurities Detection in Intensity Inhomogeneous Edible Bird’s Nest (EBN) Using a U-Net Deep Learning Model”, Int. j. eng. technol. innov., vol. 11, no. 2, pp. 135–145, Apr. 2021. https://doi.org/10.46604/ijeti.2021.6891
2. Ying Heng Yeo and Kin Sam Yen "Development of a hybrid autoencoder model for automated edible bird’s nest impurities inspection," Journal of Electronic Imaging 31(5), 051603 (2 June 2022). https://doi.org/10.1117/1.JEI.31.5.051603

**Usage**

Two model architectures were included
1. unetSM.py - a classic U-Net model for multi-class image segmentation, designed to label each pixel in the image using a single 4-channel input (RGB + White), where each pixel is classified into one of several classes.
2. ae2.py - a custom hybrid U-Net–style convolutional autoencoder with greedy training and manual control logic for pixel-wise anomaly segmentation (normal vs abnormal).

**License**

MIT License — free to use with proper attribution.

**Author**

yenksam | https://github.com/yenksam
