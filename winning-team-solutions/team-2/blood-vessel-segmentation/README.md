1. The hardware you used: CPU specs, number of CPU cores, memory, GPU specs, number of GPUs.

CPU: Core i7-13700
Memory: 96GB
GPU: NVIDIA GeForce RTX4090 × 1

2. OS/platform you used, including version number.

Arch Linux

3. Any necessary 3rd-party software, including version numbers, and installation steps. This can be provided as a Dockerfile instead of as a section in the readme.

funcy 2.0
matplotlib 3.8.2
numpy 1.26.4
opencv-python 4.9.0
pandas 1.5.3
scikit_learn 1.4.0
tensorflow 2.13

4. How to train your model

$ cd ./src
$ python create_volumetric_image.py
$ python create_sample_data.py
$ python train_0.py

5. How to make predictions on a new test set.

$ cd ./src
$ python submit.py

6. Important side effects of your code. For example, if your data processing code overwrites the original data.

Nothing.

7. Key assumptions made by your code. For example, if the outputs folder must be empty when starting a training run.

Nothing.
