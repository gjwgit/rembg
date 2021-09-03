import os
import io
import random
import numpy as np
from src.rembg.bg import remove
from PIL import Image, ImageFile
from mlhub.pkg import mlask, mlcat, get_cmd_cwd
from mlhub.utils import get_package_dir
ImageFile.LOAD_TRUNCATED_IMAGES = True


mlcat("rembg", """\
rembg is a Python library for static image background removal.\n
The backbone network based on is u2net.\n
To use this library, extra pretrained model for u2net will be downloaded.\n
See https://arxiv.org/pdf/2005.09007.pdf for paper\n
Or https://github.com/xuebinqin/U-2-Net for detailed model descriptions.
""")
mlask(end='\n')

examples = ['animal-1', 'animal-2', 'animal-3',
            'car-1', 'car-2', 'car-3',
            'food-1',
            'girl-1', 'girl-2', 'girl-1']

input_files = random.sample(examples, 3)
output_files = [x+"-out" for x in input_files]

# Creating output folder in current working directory

if not os.path.exists(os.path.join("rembg-output")):
    os.mkdir("rembg-output")
    mlcat(text="Output folder has been created in current working directory\n")

# Removal Example 1: Performing Regular Keying

mlcat("Removing Example 1", "Basic background removal using basic example")
mlask(end="\n", prompt="Press Enter to perform removal on " + input_files[0] + ".jpg")

f = np.fromfile(os.path.join(get_package_dir(), 'examples', input_files[0] + ".jpg"))
result = remove(f)
img = Image.open(io.BytesIO(result)).convert("RGBA")

f.tofile(os.path.join(get_cmd_cwd(), 'rembg-output', input_files[0] + ".jpg"))
img.save(os.path.join(get_cmd_cwd(), 'rembg-output', output_files[0] + ".png"))

mlask(end="\n", prompt="Please find the picture in the output directory, named " + output_files[0] + ".png")
del f, img, result


# Removal Example 2: Performing Regular Keying

mlcat("Removing Example 2", "Another example on basic background removal")
mlask(end="\n", prompt="Press Enter to perform removal on " + input_files[1] + ".jpg")

f = np.fromfile(os.path.join(get_package_dir(), 'examples', input_files[1] + ".jpg"))
result = remove(f)
img = Image.open(io.BytesIO(result)).convert("RGBA")

f.tofile(os.path.join(get_cmd_cwd(), 'rembg-output', input_files[1] + ".jpg"))
img.save(os.path.join(get_cmd_cwd(), 'rembg-output', output_files[1] + ".png"))

mlask(end="\n", prompt="Please find the picture in the output directory, named "+output_files[1] + ".png")
del f, img, result


# Removal Example 3: Performing Alpha-matting Keying

mlcat("Removing Example 3", '''
Alpha matting background removal using basic example.\n
Naive background removal will be used if PyMatting is not available
''')
mlask(end="\n", prompt="Press Enter to perform alpha matting removal on " + input_files[2] + ".jpg")

f = np.fromfile(os.path.join(get_package_dir(), 'examples', input_files[2] + ".jpg"))

# If alpha-matting is not available, naive keying will be used instead
result = remove(f, alpha_matting=True)
img = Image.open(io.BytesIO(result)).convert("RGBA")

f.tofile(os.path.join(get_cmd_cwd(), 'rembg-output', input_files[2] + ".jpg"))
img.save(os.path.join(get_cmd_cwd(), 'rembg-output', 'alpha-' + output_files[2] + ".png"))

mlask(end="\n", prompt="Please find the picture in the output directory, named "+output_files[2] + "-alpha.png")
del f, img, result

