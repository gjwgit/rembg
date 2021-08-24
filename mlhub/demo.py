import os
import io
import random
import numpy as np
from src.rembg.bg import remove
from PIL import Image, ImageFile
from mlhub.pkg import mlask, mlcat
from mlhub.utils import get_package_dir
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():

    mlcat("rembg", """\
    rembg is a Python library for static image background removal.
    The backbone network based on is u2net.
    To use this library, extra pretrained model for u2net will be downloaded.
    See https://arxiv.org/pdf/2005.09007.pdf for paper or https://github.com/xuebinqin/U-2-Net for detailed model descriptions.
    """)
    mlask(end='\n')

    examples = ['animal-1.jpg', 'animal-2.jpg', 'animal-3.jpg',
                'car-1.jpg', 'car-2.jpg', 'car-3.jpg',
                'food-1.jpg',
                'girl-1.jpg','girl-2.jpg','girl-1.jpg']

    input_file = random.choice(examples)
    output_file = 'out.png'

    # Removal Example 1: Performing Regular Keying

    mlcat("Removing Example 1", "Basic background removal using basic example")
    mlask(end="\n", prompt="Press Enter to perform removal on" + input_file)

    f = np.fromfile(get_package_dir() + '\\examples\\' + input_file)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(os.getcwd() + '\\' + output_file)

    mlask(end="\n", prompt="Please find the output in current working directory, named "+output_file)
    del f, img, result

    # Removal Example 2: Performing Alpha-matting Keying

    mlcat("Removing Example 2", "Alpha matting background removal using basic example")
    mlask(end="\n", prompt="Press Enter to perform alpha matting removal on" + input_file)

    f = np.fromfile(get_package_dir() + '\\examples\\' + input_file)
    result = remove(f, alpha_matting=True)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(os.getcwd() + '\\alpha-' + output_file)

    mlask(end="\n", prompt="Please find the output in current working directory, named alpha-"+output_file)
    del f, img, result


if __name__ == "main":
    main()
