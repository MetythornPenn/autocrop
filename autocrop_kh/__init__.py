import os
import gc
import cv2
import numpy as np
import onnxruntime as ort
import warnings
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress ONNX Runtime warnings by setting environment variables
os.environ["ORT_LOGGING_LEVEL"] = "3"  # Set to 3 to suppress warnings and only show errors

DEFAULT_MODEL_FILENAME = "autocrop_model_v2.onnx"
DEFAULT_HF_REPO = "metythorn/autocrop"
ENV_MODEL_DIR = "AUTOCROP_KH_MODEL_DIR"
ENV_HF_REPO = "AUTOCROP_KH_HF_REPO"


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()

def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)

def image_preproces_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def _transform(image):
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        return np.transpose(image, (2, 0, 1))

    return _transform

def _default_model_dir():
    env_dir = os.environ.get(ENV_MODEL_DIR)
    if env_dir:
        return env_dir
    return os.path.join(os.path.expanduser("~"), ".cache", "autocrop_kh")

def _hf_model_url(filename):
    repo = os.environ.get(ENV_HF_REPO, DEFAULT_HF_REPO)
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"

def _download_model(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".download"
    try:
        with urlrequest.urlopen(url) as response, open(tmp_path, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp_path, dest_path)
    except (HTTPError, URLError) as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to download model from {url}: {exc}") from exc

def ensure_model_path(model_path):
    if model_path is None:
        model_path = os.path.join(_default_model_dir(), DEFAULT_MODEL_FILENAME)

    if not model_path.endswith(".onnx"):
        raise ValueError("Unsupported model format. Only .onnx is supported.")

    if not os.path.exists(model_path):
        url = _hf_model_url(os.path.basename(model_path))
        _download_model(url, model_path)

    return model_path

def load_autocrop_model(checkpoint_path, device):
    checkpoint_path = ensure_model_path(checkpoint_path)

    providers = ort.get_available_providers()
    use_cuda = (device == 'cuda') and ('CUDAExecutionProvider' in providers)
    session = ort.InferenceSession(
        checkpoint_path,
        providers=['CUDAExecutionProvider' if use_cuda else 'CPUExecutionProvider'],
    )
    return session

preprocess_transforms = image_preproces_transforms()

def extract(image_true=None, trained_model=None, image_size=384, BUFFER=10):
    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image_true.shape
    image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE

    image_model = preprocess_transforms(image_model)
    image_model = np.expand_dims(image_model, axis=0)
    input_name = trained_model.get_inputs()[0].name
    output = trained_model.run(None, {input_name: image_model})
    out = output[0]

    del image_model
    gc.collect()

    out = np.argmax(out, axis=1, keepdims=True).transpose(0, 2, 3, 1)[0].squeeze().astype(np.int32)
    r_H, r_W = out.shape

    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half: half + IMAGE_SIZE, half: half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()

    del _out_extended
    gc.collect()

    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)
    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half
    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):
        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER

        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER

        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER

        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER

        image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image_true.dtype)
        image_extended[top_pad: top_pad + imH, left_pad: left_pad + imW, :] = image_true
        image_extended = image_extended.astype(np.float32)

        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        image_true = image_extended

    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

    final = cv2.warpPerspective(image_true, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
    final = np.clip(final, a_min=0., a_max=255.)

    return final

def autocrop(img_path=None, np_image=None, pil_image=None, model_path=None, device=None, trained_model=None, output_path=None):
    if trained_model is None:
        trained_model = load_autocrop_model(checkpoint_path=model_path, device=device)

    if img_path:
        # If img_path is provided, read the image from disk
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # Convert BGR to RGB
    elif np_image is not None:
        # If np_image is provided, use it directly
        image = np_image  # Assuming np_image is already a NumPy array in RGB format
    elif pil_image is not None:
        # If pil_image is provided, convert it to NumPy array
        image = np.array(pil_image)  # Convert PIL image to NumPy array
    else:
        raise ValueError("No image input provided. Please provide img_path, np_image, or pil_image.")

    # Perform document extraction
    extracted_image = extract(image_true=image, trained_model=trained_model)

    if output_path:
        cv2.imwrite(output_path, extracted_image[:, :, ::-1])

    return extracted_image
