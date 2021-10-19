import os
import io
import random
import numpy as np
import warnings
from src.rembg.bg import remove
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
Or https://github.com/xuebinqin/U-2-Net for detailed model descriptions.
""")
mlask(end='\n')

examples = ['animal-1', 'animal-2', 'animal-3',
            'car-1', 'car-2', 'car-3',
            'food-1',
            'girl-1', 'girl-2', 'girl-1']

input_files = random.sample(examples, 3)

save_fig = yes_or_no("Do you want to save the results under current working directory?",
                     yes=False)

# Removal Example 1: Performing Regular Keying

mlcat("Removing Example 1", "Basic background removal using basic example.")
mlask(end="\n", prompt="Press Enter to perform removal on " + input_files[0] + ".jpg")

f = np.fromfile(os.path.join(get_package_dir(), 'examples', input_files[0] + ".jpg"))
result = remove(f)
f = Image.open(io.BytesIO(f)).convert("RGBA")
img = Image.open(io.BytesIO(result)).convert("RGBA")
fig, plot = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
plot[0].imshow(f)
plot[0].set_title('Original Image')
plot[0].axis('off')
plot[1].imshow(img)
plot[1].set_title('Removal Result')
plot[1].axis('off')
fig.suptitle('Removal Example 1 on '+ input_files[0] + ".jpg")
if save_fig:
    plt.savefig(os.path.join(get_cmd_cwd(), input_files[0]+'.out.png'))
else:
    plt.show()
del f, img, result, fig, plot
print("Background removal on " + input_files[0] + ".jpg has completed.\n")

# Removal Example 2: Performing Regular Keying

mlcat("Removing Example 2", "Another example on basic background removal.")
mlask(end="\n", prompt="Press Enter to perform removal on " + input_files[1] + ".jpg")

f = np.fromfile(os.path.join(get_package_dir(), 'examples', input_files[1] + ".jpg"))
result = remove(f)
f = Image.open(io.BytesIO(f)).convert("RGBA")
img = Image.open(io.BytesIO(result)).convert("RGBA")
fig, plot = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
plot[0].imshow(f)
plot[0].set_title('Original Image')
plot[0].axis('off')
plot[1].imshow(img)
plot[1].set_title('Removal Result')
plot[1].axis('off')
fig.suptitle('Removal Example 2 on '+ input_files[1] + ".jpg")
if save_fig:
    plt.savefig(os.path.join(get_cmd_cwd(), input_files[1]+'.out.png'))
else:
    plt.show()
del f, img, result, fig, plot
print("Background removal on " + input_files[1] + ".jpg has completed.\n")

# Removal Example 3: Performing Alpha-matting Keying

mlcat("Removing Example 3", '''Alpha matting background removal using basic example.\n
Naive background removal will be used if PyMatting is not available.
''')
mlask(end="\n",
      prompt="Press Enter to perform alpha matting removal on " + input_files[2] + ".jpg")

f = np.fromfile(os.path.join(get_package_dir(), 'examples', input_files[2] + ".jpg"))

# If alpha-matting is not available, naive keying will be used instead
result = remove(f, alpha_matting=True)
f = Image.open(io.BytesIO(f)).convert("RGBA")
img = Image.open(io.BytesIO(result)).convert("RGBA")
fig, plot = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
plot[0].imshow(f)
plot[0].set_title('Original Image')
plot[0].axis('off')
plot[1].imshow(img)
plot[1].set_title('Alpha Removal Result')
plot[1].axis('off')
fig.suptitle('Alpha Matting Removal Example on '+ input_files[2] + ".jpg")
if save_fig:
    plt.savefig(os.path.join(get_cmd_cwd(), input_files[2]+'.out.png'))
else:
    plt.show()
del f, img, result, fig, plot
print("Alpha matting removal on " + input_files[2] + ".jpg has completed.\n")

if save_fig:
    mlask(end="\n",
          prompt="Please refer to current working directory to find the outputs")
