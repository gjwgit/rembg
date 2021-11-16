import os
import io
import random
import numpy as np
import warnings
import filetype
from src.rembg.bg import remove, video_remove, alpha_layer_remove
from PIL import Image, ImageFile
from mlhub.pkg import mlask, mlcat, get_cmd_cwd
from mlhub.utils import get_package_dir, yes_or_no
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

mlcat("rembg", """\
rembg is a Python library for static image background removal.\n
This library based on u2net as backbone network.\n
To use this library, extra pretrained model for u2net will be downloaded.\n
See https://arxiv.org/pdf/2005.09007.pdf for paper\n
Or https://github.com/xuebinqin/U-2-Net for detailed model descriptions.\n
This wizard will run through the basic functions provided by the library.
""")
mlask(end='\n')

print("Type the path to your input file, or remain blank if you want to use provided example")
input_file = input()
if input_file:
    examples = ['animal-1', 'animal-2', 'animal-3',
                'car-1', 'car-2', 'car-3',
                'food-1',
                'girl-1', 'girl-2', 'girl-1']
    input_file = random.sample(examples, 1)[0]
    print("No valid input file is provided. The demo will use " + input_file + ".jpg as the input")

save_flag = yes_or_no("Do you want to save the results under current working directory?\n"
                      "You can always specify such path using -o parameter in CLI.",
                      yes=False)

if save_flag:
    output_path = os.path.join(get_cmd_cwd())
else:
    print("Please input a valid directory for saving the result:\n")
    output_path = input()

if os.path.exists(input_file) \
   and filetype.guess(input_file).mime.find('image') >= 0:
    f = np.fromfile(input_file)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    comparison_flag = yes_or_no("Do you want to save the results with its input as comparison?\n"
                                "You can always choose to do so by adding -c parameter in CLI",
                                yes=False)
    if comparison_flag:
        f = Image.open(io.BytesIO(f)).convert("RGBA")
        fig, plot = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        plot[0].imshow(f)
        plot[0].set_title('Original Image')
        plot[0].axis('off')
        plot[1].imshow(img)
        plot[1].set_title('Removal Result')
        plot[1].axis('off')
        fig.suptitle('Removal Result')
    else:
        plt.axis('off')
        plt.imshow(img)

elif os.path.exists(input_file) \
     and filetype.guess(input_file).mime.find('video') >= 0:
    if not os.path.exists(output_path):
        print("You have to provide a valid directory for saving the result:\n")
        output_path = input()
        _, output_file = os.path.split(input_file)
        output_file = output_file.split('.')
    flag = video_remove(input_file, os.path.join(output_path, output_file[0]+'.out.'+output_file[1]))


