import cv2
import numpy as np

hdr_image_path = "quarry_03_1k.hdr"
img = cv2.imread(hdr_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
if img is None:
    raise RuntimeError(f"Failed to load {hdr_image_path}")

# Compute per-pixel max across channels (brightness)
pixel_brightness = np.max(img, axis=2)
max_brightness = np.max(pixel_brightness)

# Find pixel(s) with this max brightness
brightest_mask = (pixel_brightness == max_brightness)

# Prepare an output image of zeros
output_img = np.zeros_like(img)

# For each channel, set the value only at the brightest pixel(s)
for c in range(3):
    output_img[:,:,c][brightest_mask] = img[:,:,c][brightest_mask]

# Save the result
output_path = "quarry_03_1k_brightest_pixel_all_channels.hdr"
cv2.imwrite(output_path, output_img)
print(f"Saved HDR with only the brightest pixel (all channels kept) to: {output_path}")
