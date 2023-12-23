# NeFour

Neural Implicit Fourier Transform for Remote Sensing Shadow Removal.

This is the code of the implementation of the NeFour.

# Training
1. Hyperparameter (.src/train.py)
3. Python train.py

# Testing
1. Put the testing data to corresponding folders (hazy image to ./data/test_data/input, GT to ./data/test_data/target, GT for full-reference evaluation, such as PSNR and SSIM)
2. Python Enh_eval.py
3. Find the result in corresponding folder (./checkpoints/XX/test_results)

# Remote Sensing Image Dehazing Dataset (RSID)
Download RSID from Baidu Cloud: https://pan.baidu.com/s/1zzk1KiKJHnZPHg4BV5U7dA?pwd=1004 key: 1004 or Google Drive: https://drive.google.com/file/d/1FC7oSkGTthjHl2sKN-yGrKhssgV0QB4F/view?usp=sharing

# Natural Image Dehazing Dataset (NID)
Download NID from Baidu Cloud: https://pan.baidu.com/s/1bvXiWE3kVH_xhISL_SJ6xA?pwd=1004 key: 1004 or Google Drive: https://drive.google.com/file/d/1vyGsFDaV9uVMO4Qeg1dRYitbIDYSC_eX/view?usp=sharing

# Contact Us
If you have any questions, please contact us (chikaichen@mail.nwpu.edu.cn).

# Acknowledgments
Code is implemented based on https://github.com/xw-hu/Mask-ShadowGAN.

Metric is implemented based on https://ieeexplore.ieee.org/document/7300447 (UCIQE) and https://github.com/imfing/CEIQ (CEIQ).
