import functools
import io
import numpy as np
from PIL import Image
from .u2net import detect
from pickle import UnpicklingError
import cv2


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
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
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
    img = Image.open(io.BytesIO(data)).convert("RGB")
    mask = detect.predict(model, np.array(img)).convert("L")
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
    return bio.getbuffer()


def extract_frame(file_path):
    cap = cv2.VideoCapture(file_path)
    flag, img = cap.read()
    while flag:
        yield img
        flag, img = cap.read()
    cap.release()


def video_remove(
    data,
    output_path,
    model_name="u2net",
    alpha_matting=False,
    *args, **kwargs
):
    model = get_model(model_name)

    cap = cv2.VideoCapture(data)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video = cv2.VideoWriter(output_path, fourcc, fps, (width,height))

    for img in extract_frame(file_path=data):
        mask = detect.predict(model, np.array(img)).convert("L")
        cutout = None
        if alpha_matting:
            cutout = alpha_matting_cutout(
                img,
                mask,
                *args, **kwargs
            )
        if cutout is None:
            cutout = naive_cutout(img, mask)
        video.write(cutout)
    video.release()
    return True
