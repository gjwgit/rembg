import argparse
import os
import warnings
import filetype
from src.rembg.bg import portrait, video_portrait
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from mlhub.utils import get_package_dir
from mlhub.pkg import get_cmd_cwd

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

model_path = os.environ.get(
    "U2NET_PATH",
    os.path.expanduser(os.path.join(get_package_dir(), "model")),
)

ap = argparse.ArgumentParser()

ap.add_argument(
    "input",
    nargs="?",
    type=str,
    help="Path to the input image.",
)

ap.add_argument(
    "-o",
    "--output",
    nargs="?",
    type=str,
    help="Path to the output png image.",
)

ap.add_argument(
    '-c',
    '--composite',
    action='store_true',
    help="Display both original and result picture"
)

ap.add_argument(
    "-cs",
    "--composite-sigma",
    default=2,
    type=float,
    help="Sigma value used for Gaussian filters when compositing.",
)

ap.add_argument(
    "-ca",
    "--composite-alpha",
    default=0.5,
    type=float,
    help="Alpha value used for Gaussian filters when compositing.",
)

args = ap.parse_args()

if args.input is None:
    raise FileNotFoundError("Please specify a valid image file as input")

if os.path.isabs(args.input):
    input_path = args.input
else:
    input_path = os.path.join(get_cmd_cwd(), args.input)

if args.output is not None:
    if os.path.isabs(args.output):
        output_path = args.output
    else:
        output_path = os.path.join(get_cmd_cwd(), args.output)

if os.path.exists(input_path) \
    and filetype.guess(input_path).mime.find('image') >= 0:
    f = Image.open(input_path).convert("RGB")
    result = portrait(
        f,
        model_name='u2net_portrait',
        composite=args.composite,
        sigma=args.composite_sigma,
        alpha=args.composite_alpha
    )
    plt.axis('off')
    plt.imshow(result)

    if args.output is None:
        output_path, output_file = os.path.split(input_path)
        output_file = output_file.split('.')
        plt.savefig(os.path.join(output_path, output_file[0] + '_out.jpg'))
    else:
        output_dir, _ = os.path.split(output_path)
        if output_dir != '' and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path)

elif os.path.exists(input_path) \
    and filetype.guess(input_path).mime.find('video') >= 0:
    if not os.path.exists(os.path.split(output_path)[0]):
        raise FileNotFoundError("You have to specific a valid output path for a video input")
    else:
        flag = video_portrait(input_path, output_path)

else:
    raise FileNotFoundError("The input " + input_path + " is not a valid path to a image file")
