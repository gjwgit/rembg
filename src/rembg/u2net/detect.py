import errno
import os
import sys

import numpy as np
import requests
import torch
from torch.autograd import Variable
from hsh.library.hash import Hasher
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from . import data_loader, u2net


def download_file_from_google_drive(id, fname, destination):
    print("Downloading "+ fname)

    head, tail = os.path.split(destination)
    os.makedirs(head, exist_ok=True)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    total = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as file, tqdm(
        desc=f"Downloading {tail} to {head}",
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def load_model(model_name: str = None):
    if model_name is None:
        model_name = 'u2net'
    hasher = Hasher()

    if not os.path.exists(os.path.join("model")):
        os.mkdir("model")

    path = os.environ.get(
        "U2NET_PATH",
        os.path.expanduser(os.path.join("model", model_name + ".pth")),
    )

    if model_name == "u2netp":
        net = u2net.U2NETP(3, 1)
        if (
            not os.path.exists(path)
            or hasher.md5(path) != "e4f636406ca4e2af789941e7f139ee2e"
        ):
            download_file_from_google_drive(
                "1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
                "u2netp.pth",
                path,
            )

    elif model_name == "u2net":
        net = u2net.U2NET(3, 1)
        if (
            not os.path.exists(path)
            or hasher.md5(path) != "347c3d51b01528e5c6c071e3cff1cb55"
        ):
            download_file_from_google_drive(
                "1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
                "u2net.pth",
                path,
            )

    elif model_name == "u2net_human_seg":
        net = u2net.U2NET(3, 1)
        if (
            not os.path.exists(path)
            or hasher.md5(path) != "09fb4e49b7f785c9f855baf94916840a"
        ):
            download_file_from_google_drive(
                "1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P",
                "u2net_human_seg.pth",
                path,
            )
    elif model_name == "u2net_portrait":
        net = u2net.U2NET(3, 1)
        if (
            not os.path.exists(path)
            or hasher.md5(path) != 'c7cff57409664b679dc7a5a445b259e4'
        ):
            download_file_from_google_drive(
                "1IG3HdpcRiDoWNookbncQjeaPN28t90yW",
                "u2net_portrait.pth",
                path,
            )
    else:
        print("Choose between u2net, u2net_human_seg or u2netp", file=sys.stderr)
    print(model_name)
    try:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(path))
            net.to(torch.device("cuda"))
        else:
            net.load_state_dict(
                torch.load(
                    path,
                    map_location="cpu",
                )
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), model_name + ".pth"
        )

    net.eval()

    return net


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose(
        [data_loader.RescaleT(320), data_loader.ToTensorLab(flag=0)]
    )
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


def predict(net, item, portrait=False):
    sample = preprocess(item)

    with torch.no_grad():

        if torch.cuda.is_available():
            inputs_test = torch.cuda.FloatTensor(
                sample["image"].unsqueeze(0).cuda().float()
            )
        else:
            inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        if portrait:
            pred = 1.0 - d1[:, 0, :, :]
        else:
            pred = d1[:, 0, :, :]
        predict = norm_pred(pred)

        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        # img = Image.fromarray(predict_np * 255).convert("RGB")

        del d1, d2, d3, d4, d5, d6, d7, pred, predict, inputs_test, sample

        return predict_np


def inference(net,input):
    # Deprecated
    # # normalize the input
    # tmpImg = np.zeros((input.shape[0],input.shape[1],3))
    # input = input/np.max(input)
    #
    # tmpImg[:,:,0] = (input[:,:,2]-0.406)/0.225
    # tmpImg[:,:,1] = (input[:,:,1]-0.456)/0.224
    # tmpImg[:,:,2] = (input[:,:,0]-0.485)/0.229
    #
    # # convert BGR to RGB
    # tmpImg = input.transpose((2, 0, 1))
    # tmpImg = tmpImg[np.newaxis,:,:,:]
    # tmpImg = torch.from_numpy(tmpImg)
    #
    # # convert numpy array to torch tensor
    # print(tmpImg.shape)
    # tmpImg = tmpImg.type(torch.FloatTensor)
    pass
