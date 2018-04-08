# deeproad
## CVPR 2018 DeepGlobe Road Extraction Challenge

This repo contains the src for my submission to the [CVPR 2018 DeepGlobe Road Extraction Challenge](http://deepglobe.org/). These scripts implement a model using convolutional architectures to estimate the pixel-wise probabilities whether they are part of a road. This problem can also be described as an image segmentation task     

### Directory Structure

The repo contains one file to specify the architecture and perform training as well as two other modules for evaluation and utilities. The repo also contains a very small sample of images and masks as an example of the training/evaluation data.

    |
    |-- deepglobe.py
    |-- deepglobe_eval.py
    |-- utils.py
    |-- data
         |
         |-- train
         |     |
         |     |-- X
         |     |   |-- img
         |     |        |
         |     |        |-- *_sat.jpg
         |     |-- y
         |         |-- mask
         |              |
         |              |-- *_mask.png
         |-- valid
         |-- test
         
### Training
 
To train this network, specify the following parameters
1. NUM_EPOCHS
2. BATCH_SIZE
3. TRAIN_DIR
4. VALID_DIR
5. OUT_DIR
6. prep (default=False)
7. seed (default=1)
   
    python deepglobe.py 10 16 /data/deepglobe/road/train/ /data/deepglobe/road/valid/ out/
