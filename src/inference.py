import cv2
import onnxruntime as ort
from autocrop_kh import autocrop


img_path = "sample/img-1.jpg"
output_path = "sample/result-img-1.png"
model_path = None 

device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"

cropped_image = autocrop(
    img_path=img_path,
    model_path=model_path,
    device=device,
    output_path=output_path,
)
print(f"Extracted document saved to {output_path}")
