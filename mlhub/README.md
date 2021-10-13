# rembg
rembg is a Python library for static image background removal.
This library based on u2net as backbone network.
To use this library, extra pretrained model for u2net will be downloaded.

See https://arxiv.org/pdf/2005.09007.pdf for paper or https://github.com/xuebinqin/U-2-Net for detailed model descriptions.

This repo is adapted from the [source library](https://github.com/danielgatis/rembg). Some unused codes and dependencies have been removed

# Usage
- To install MLHub on Ubuntu
```shell
$ pip3 install mlhub
$ ml configure
```

- To install rembg using MLHub
```shell
$ ml install rembg
```

- To run rembg demo using provided examples
```shell
$ ml demo rembg
```
Pretrained U-2-Net model will be downloaded on first use

- To perform removal on a custom image
```shell
$ ml cutout rembg PATH_TO_INPUT_FILE [-o PATH_TO_OUTPUT_FILE]
```
Replace PATH_TO_INPUT_FILE and PATH_TO_OUTPUT_FILE with your corresponding path.
The output path is optional here. If unspecified, the output file will be generated in the input folder.

- To perform removal on a custom image with comparison to its original image
```shell
$ ml cutout rembg PATH_TO_INPUT_FILE -c
```
Replace PATH_TO_INPUT_FILE with your corresponding path

- To perform removal on a custom image using alpha-matting
```shell
$ ml cutout rembg PATH_TO_INPUT_FILE -o PATH_TO_OUTPUT_FILE -a True
```
Replace PATH_TO_INPUT_FILE and PATH_TO_OUTPUT_FILE with your corresponding path

- Full usage of cutout.py
```shell
usage: cutout.py [-h] [-o [OUTPUT]] [-m {u2net,u2netp,u2net_human_seg}] [-c] [-j] [-a [ALPHA_MATTING]]
                 [-af ALPHA_MATTING_FOREGROUND_THRESHOLD] [-ab ALPHA_MATTING_BACKGROUND_THRESHOLD]
                 [-ae ALPHA_MATTING_ERODE_SIZE] [-az ALPHA_MATTING_BASE_SIZE]
                 [input]

positional arguments:
  input                 Path to the input image.

optional arguments:
  -h, --help            show this help message and exit
  -o [OUTPUT], --output [OUTPUT]
                        Path to the output png image.
  -m {u2net,u2netp,u2net_human_seg}, --model {u2net,u2netp,u2net_human_seg}
                        The model name.
  -c, --compare         Display both original and result picture
  -j, --jpeg            Store/Display the file in JPEG format without transparent layer
  -a [ALPHA_MATTING], --alpha-matting [ALPHA_MATTING]
                        When true use alpha matting cutout.
  -af ALPHA_MATTING_FOREGROUND_THRESHOLD, --alpha-matting-foreground-threshold ALPHA_MATTING_FOREGROUND_THRESHOLD
                        The trimap foreground threshold.
  -ab ALPHA_MATTING_BACKGROUND_THRESHOLD, --alpha-matting-background-threshold ALPHA_MATTING_BACKGROUND_THRESHOLD
                        The trimap background threshold.
  -ae ALPHA_MATTING_ERODE_SIZE, --alpha-matting-erode-size ALPHA_MATTING_ERODE_SIZE
                        Size of element used for the erosion.
  -az ALPHA_MATTING_BASE_SIZE, --alpha-matting-base-size ALPHA_MATTING_BASE_SIZE
                        The image base size.
```
