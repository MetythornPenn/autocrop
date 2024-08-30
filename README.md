# Autocrop

#### Automatic Document Segmentation and Cropping for Khmer IDs, Passport and Documents

Autocrop is a Python package for automatic document segmentation and cropping, with a focus on Khmer IDs, Passport and other documents. It uses a DeepLabV3 model training on Khmer ID, Passport document datasets to accurately segment and extract documents from images.

License: [Apache-2.0 License](https://github.com/MetythornPenn/sdab/blob/main/LICENSE)

## Installation

#### Install from PyPI
```sh
pip install autocrop
```

#### Install from source

```sh

# clone repo 
git clone https://github.com/MetythornPenn/autocrop.git

# install lib from source
pip install -e .
```

## Usage

#### Download sample image

```bash
# Download sample image
wget -O img-1.jpg https://github.com/MetythornPenn/autocrop/blob/main/sample/img-1.jpg

# Download model from here
wget -O autocrop_model_mbv3.pth https://github.com/MetythornPenn/autocrop/blob/main/models/autocrop_model_mbv3.pth

```

#### Python API

```python
import torch
import cv2
import requests

from autocrop import autocrop

# Function to download files using requests
def download_file(url, output_path):
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors
    with open(output_path, 'wb') as file:
        file.write(response.content)

# URLs for the image and model
img_url = "https://github.com/MetythornPenn/autocrop/raw/main/sample/img-1.jpg"
model_url = "https://github.com/MetythornPenn/autocrop/raw/main/models/autocrop_model_mbv3.pth"

# Local paths to save the files
img_path = "img-1.jpg"
model_path = "autocrop_model_mbv3.pth"

# Download the image and model files
download_file(img_url, img_path)
download_file(model_url, model_path)

# Specify device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Perform document extraction
extracted_document = autocrop(img_path, model_path, device)

# Save the extracted document
output_path = "extracted_document.jpg"
cv2.imwrite(output_path, extracted_document[:, :, ::-1])  # Convert back to BGR for saving

print(f"Extracted document saved to {output_path}")

# Display the result (Optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.imshow(extracted_document / 255.0)
plt.title("Extracted Document")
plt.show()


```

- `img_path`: Path of the input image file.
- `model_path`: Path to the pre-trained model (local path).
- `device`: Specify `cpu` or `cuda` (default is `gpu`).
- `output_path`: Path where the extracted document image will be saved.

## Reference 
- Inspired by [DeepLabV3](https://paperswithcode.com/method/deeplabv3)