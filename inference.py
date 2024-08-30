import torch 
import cv2
# import matplotlib
from autocrop import autocrop 


img_path = "sample/img-1.jpg"
output_path = "extracted_document.jpg"
model_path = "models/autocrop_model_mbv3.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


autocrop = autocrop(img_path, model_path, device)


cv2.imwrite(output_path, autocrop[:, :, ::-1])  # Convert back to BGR for saving

# Display the result
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.imshow(autocrop / 255.0)
plt.title("Extracted Document")
plt.show()

print(f"Extracted document saved to {output_path}")



