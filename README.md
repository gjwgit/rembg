# rembg
rembg is a Python library for static image background removal.
The backbone network based on is u2net.
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

- To key a custom image
```shell
$ ml keying rembg PATH_TO_INPUT_FILE -o PATH_TO_OUTPUT_FILE
```
Replace PATH_TO_INPUT_FILE and PATH_TO_OUTPUT_FILE with your corresponding path

- To key a custom image using alpha-matting
```shell
$ ml keying rembg PATH_TO_INPUT_FILE -o PATH_TO_OUTPUT_FILE -a True
```
Replace PATH_TO_INPUT_FILE and PATH_TO_OUTPUT_FILE with your corresponding path
