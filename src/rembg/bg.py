import functools
import io
import numpy as np
from PIL import Image
from skimage import transform
from skimage.filters import gaussian
from .u2net import detect
from pickle import UnpicklingError


def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold=240,
    background_threshold=10,
    erode_structure_size=10,
    base_size=1000,
):
    try:
        from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
        from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
        from pymatting.util.util import stack_images
        from scipy.ndimage.morphology import binary_erosion
    except:
        print("PyMatting seems unavailable currently.\nCheck your environment or disable alpha-matting")
        return None
    else:
        size = img.size

        img.thumbnail((base_size, base_size), Image.LANCZOS)
        mask = mask.resize(img.size, Image.LANCZOS)

        img = np.asarray(img)
        mask = np.asarray(mask)

        # guess likely foreground/background
        is_foreground = mask > foreground_threshold
        is_background = mask < background_threshold

        # erode foreground/background
        structure = None
        if erode_structure_size > 0:
            structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int)

        is_foreground = binary_erosion(is_foreground, structure=structure)
        is_background = binary_erosion(is_background, structure=structure, border_value=1)

        # build trimap
        # 0   = background
        # 128 = unknown
        # 255 = foreground
        trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
        trimap[is_foreground] = 255
        trimap[is_background] = 0

        # build the cutout image
        img_normalized = img / 255.0
        trimap_normalized = trimap / 255.0

        alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
        foreground = estimate_foreground_ml(img_normalized, alpha)
        cutout = stack_images(foreground, alpha)

        cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
        cutout = Image.fromarray(cutout)
        cutout = cutout.resize(size, Image.LANCZOS)

        return cutout


def naive_cutout(img, mask):
    img_shape = img.size if type(img) == Image.Image else img.shape[0:2]
    empty = Image.new("RGBA", img_shape, 0)
    cutout = Image.composite(img,
                             empty,
                             mask.resize(img_shape, Image.LANCZOS))
    return cutout


@functools.lru_cache(maxsize=None)
def get_model(model_name):
    error_count = 0
    return_model = None
    while error_count < 5:
        try:
            if model_name == "u2netp":
                return_model = detect.load_model(model_name="u2netp")
                break
            if model_name == "u2net_human_seg":
                return_model = detect.load_model(model_name="u2net_human_seg")
                break
            if model_name == "u2net_portrait":
                return_model = detect.load_model(model_name="u2net_portrait")
                break
            else:
                return_model = detect.load_model(model_name="u2net")
                break
        except (OSError, ConnectionError, UnpicklingError):
            print("Attempt "+str(error_count+1)+" failed.")
            error_count += 1
            continue
    if return_model is None:
        raise Exception("All 5 attempts seems failed.\n "
                        "Please check your Internet connection or download the weight manually and try again")
    else:
        return return_model


def remove(
    data,
    model_name="u2net",
    alpha_matting=False,
    *args, **kwargs
):
    model = get_model(model_name)
    if isinstance(data, np.ndarray):
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        img = data
    mask = Image.fromarray(detect.predict(model, np.array(img)) * 255).convert("L")
    cutout = None

    if alpha_matting:
        cutout = alpha_matting_cutout(
                img,
                mask,
                *args, **kwargs
        )

    if cutout is None:
        cutout = naive_cutout(img, mask)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    return Image.open(io.BytesIO(bio.getbuffer())).convert("RGBA")


def portrait(
    data,
    model_name='u2net_portrait',
    composite=False,
    sigma=2,
    alpha=0.5
):
    model = get_model(model_name)
    if isinstance(data, np.ndarray):
        img = data
    else:
        img = np.array(data)
    if img.shape[0] < 512 and img.shape[1] < 512:
        print("The size of the input picture is too small to generate. The result may be unexpected.")
        print("To obtain a good portrait, use a picture larger than 512*512 with a clear face.")
    output = detect.predict(model, img, True)
    if composite:
        output = transform.resize(output, img.shape[0:2],order=2)
        output = output/(np.amax(output)+1e-8) * 255
        output = output[:,:,np.newaxis]
        img_blurred = gaussian(img,sigma=sigma,preserve_range=True)
        output = img_blurred*alpha + output*(1-alpha)
        output = Image.fromarray(output.astype(np.uint8)).convert('RGB')
    else:
        output = Image.fromarray(output * 255).convert('RGB')
    return output


def extract_frame(file_path):
    # Deprecated
    try:
        import cv2
    except ModuleNotFoundError:
        print("OpenCV library is not currently installed, which is required for this functionality")
        print("Please run 'pip install opencv-python' in command-line to install dependency")
        return False
    cap = cv2.VideoCapture(file_path)
    flag, img = cap.read()
    while flag:
        yield Image.fromarray(img.astype('uint8')).convert("RGBA")
        flag, img = cap.read()
    cap.release()


def alpha_layer_remove(input_image, bg_color=np.array([255, 255, 255])):
    if isinstance(input_image, np.ndarray):
        img = input_image
    else:
        img = np.array(input_image)
    alpha = (img[:, :, 3] / 255).reshape(img.shape[:2] + (1,))
    output = bg_color * (1 - alpha) + (img[:, :, :3] * alpha)
    return output.astype(np.uint8)


def video_remove(
    input_path,
    output_path,
    model_name="u2net",
    alpha_matting=False,
    *args, **kwargs
):
    try:
        import ffmpeg
    except ModuleNotFoundError:
        print("ffmpeg library is not currently installed, which is required for this functionality")
        print("Please run 'pip install opencv-python' and 'apt install ffmpeg' in command-line to install dependency")
        return False
    probe = ffmpeg.probe(input_path)
    width = probe['streams'][0]['width']
    height = probe['streams'][0]['height']
    frame_rate = probe['streams'][0]['avg_frame_rate']
    data = ffmpeg.input(input_path)
    video_frames = np.frombuffer(data
                                 .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                                 .run(quiet=True)[0], np.uint8).reshape([-1, height, width, 3])
    output = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height)) \
                   .output(output_path, r=frame_rate).overwrite_output() \
                   .run_async(pipe_stdin=True, quiet=True)
    model = get_model(model_name)
    for i in range(video_frames.shape[0]):
        img = Image.fromarray(video_frames[i, :, :, :]).convert('RGB')
        mask = Image.fromarray(detect.predict(model, np.array(img)) * 255).convert("L")
        cutout = None
        if alpha_matting:
            cutout = alpha_matting_cutout(
                img,
                mask,
                *args, **kwargs
            )
        if cutout is None:
            cutout = naive_cutout(img, mask)
        output.stdin.write(alpha_layer_remove(cutout).tobytes())
    output.stdin.close()
    output.wait()
    return True


def video_portrait(
    input_path,
    output_path,
    model_name='u2net_portrait',
    composite=False,
    sigma=2,
    alpha=0.5):
    try:
        import ffmpeg
    except ModuleNotFoundError:
        print("ffmpeg library is not currently installed, which is required for this functionality")
        print("Please run 'pip install opencv-python' and 'apt install ffmpeg' in command-line to install dependency")
        return False
    probe = ffmpeg.probe(input_path)
    width = probe['streams'][0]['width']
    height = probe['streams'][0]['height']
    frame_rate = probe['streams'][0]['avg_frame_rate']
    data = ffmpeg.input(input_path)
    video_frames = np.frombuffer(data
                                 .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                                 .run(quiet=True)[0], np.uint8).reshape([-1, height, width, 3])
    output = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height)) \
                   .output(output_path, r=frame_rate).overwrite_output() \
                   .run_async(pipe_stdin=True, quiet=True)
    model = get_model(model_name)
    for i in range(video_frames.shape[0]):
        img = Image.fromarray(video_frames[i, :, :, :]).convert('RGB')
        result = detect.predict(model, img, True)
        if composite:
            result = transform.resize(result, img.shape[0:2], order=2)
            result = result / (np.amax(result) + 1e-8) * 255
            result = result[:, :, np.newaxis]
            img_blurred = gaussian(img, sigma=sigma, preserve_range=True)
            result = img_blurred * alpha + result * (1 - alpha)
            result = result.astype(np.uint8)
        else:
            result = (result * 255).astype(np.uint8)
        output.stdin.write(result.tobytes())
    output.stdin.close()
    output.wait()
    return True
