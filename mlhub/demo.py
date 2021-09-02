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

examples = ['animal-1.jpg', 'animal-2.jpg', 'animal-3.jpg',
            'car-1.jpg', 'car-2.jpg', 'car-3.jpg',
            'food-1.jpg',
            'girl-1.jpg', 'girl-2.jpg', 'girl-1.jpg']

input_files = random.sample(examples, 3)
output_files = ["output-"+x for x in input_files]

# Removal Example 1: Performing Regular Keying

mlcat("Removing Example 1", "Basic background removal using basic example")
mlask(end="\n", prompt="Press Enter to perform removal on " + input_files[0])

f = np.fromfile(os.path.join(get_package_dir(), 'examples', input_files[0]))
result = remove(f)
img = Image.open(io.BytesIO(result)).convert("RGBA")

f.tofile(os.path.join(get_cmd_cwd(), input_files[0]))
img.save(os.path.join(get_cmd_cwd(), output_files[0]))

mlask(end="\n", prompt="Please find the output in current working directory, named "+output_files[0])
del f, img, result

# Removal Example 2: Performing Regular Keying

mlcat("Removing Example 2", "Another example on basic background removal")
mlask(end="\n", prompt="Press Enter to perform removal on " + input_files[1])

f = np.fromfile(os.path.join(get_package_dir(), 'examples', input_files[1]))
result = remove(f)
img = Image.open(io.BytesIO(result)).convert("RGBA")

f.tofile(os.path.join(get_cmd_cwd(), input_files[1]))
img.save(os.path.join(get_cmd_cwd(), output_files[1]))

mlask(end="\n", prompt="Please find the output in current working directory, named "+output_files[1])
del f, img, result


# Removal Example 3: Performing Alpha-matting Keying

mlcat("Removing Example 3", '''
Alpha matting background removal using basic example.\n
Naive background removal will be used if PyMatting is not available
''')
mlask(end="\n", prompt="Press Enter to perform alpha matting removal on " + input_files[2])

f = np.fromfile(os.path.join(get_package_dir(), 'examples', input_files[2]))

# If alpha-matting is not available, naive keying will be used instead
result = remove(f, alpha_matting=True)
img = Image.open(io.BytesIO(result)).convert("RGBA")

f.tofile(os.path.join(get_cmd_cwd(), input_files[2]))
img.save(os.path.join(get_cmd_cwd(), 'alpha-' + output_files[2]))

mlask(end="\n", prompt="Please find the output in current working directory, named alpha-"+output_files[2])
del f, img, result

