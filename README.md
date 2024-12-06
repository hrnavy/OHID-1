# OHID-1
 We build a new set of hyperspectral data with complex characteristics using data from Orbita and named it Orbita Hyperspectral Images Dataset-1 (OHID-1). It describes different type of areas at Zhuhai City, China. 

# original data
 Baidu Netdisk： [fusioned](https://pan.baidu.com/s/1qMtY7ossLwRh0pI2v2bnDg?pwd=bi70) <https://pan.baidu.com/s/1qMtY7ossLwRh0pI2v2bnDg?pwd=bi70> code：bi70 
 
 This link provides access to the raw data and annotations of the OHID-1 dataset, which includes two different data formats: .mat and .tif. All data have a size of 5056x5056 pixels. The raw data consists of 32 bands, while the annotation data consists of 1 band.
# dataset
 The "image" folder contains 10 hyperspectral images, each with 32 spectral bands, a size of 512 × 512 pixels, and depicting 7 types of objects. The naming format is 201912_n.tif, where n ranges from 1 to 10.
 The "labels" folder contains the labels for the ten images in the "images" folder, with the same naming format of 201912_n.tif, where n also ranges from 1 to 10. Each label has values ranging from 0 to 7, and the category represented by each value can be found in sample_proportion.png.201912_n.png, 
 where n ranges from 1 to 10, represents the bar chart distribution of each category in 201912_n.tif.201912_n_color.png, where n ranges from 1 to 10, represents the visualized pseudocolor map of the labels in 201912_n.tif.

# preprocessing
 The "HSI_Classification" folder Provide the code for band synthesis and slicing of the original file.

# code
 The "HSI_Classification" folder contains the code for ID CNN, 2D CNN, 3D CNN, and SVM. These codes are built upon the [HSI_Classification](https://github.com/zhangjinyangnwpu/HSI_Classification) repository, with the primary changes made in the unit.py file.
 The "HyLiTE" files store the HyLITE code. These codes are built upon  [HyLITE](https://github.com/zhoufangqin/hylite), with the main changes made in the main.py file. We have added code for reading some additional files to make it compatible with other datasets.

_To use the code, please follow the instructions from the original project code, configure the environment, and replace the file that we have changed._
