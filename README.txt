# Image Blending using Laplacian Pyramids - README


## Overview:

This Python script performs image blending using Laplacian Pyramids. The code enables users to blend two 
images seamlessly based on specified criteria, such as the depth of the pyramid, sigma values, and masking. 
The blending is achieved through the creation of Gaussian and Laplacian pyramids, followed by a pixel-wise 
combination using a given mask.

## Dependencies:

- OpenCV (cv2)
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- SciPy (scipy.ndimage)


## Usage:

1. **Selecting an Image Region for Blending:**
    - Use the `select_image_from` function to select a region of interest (ROI) from an image.
    - Returns the clipped image
  
2. **Finding Pyramid Depth:**
    - Utilize the `find_pyramid_depth` function to determine the appropriate depth for the Laplacian pyramid.

3. **Creating Gaussian Pyramids:**
    - Use `gaussian_pyramid` and `gaussian_pyramid_color` to create Gaussian pyramids 
    for grayscale and color images, respectively.
    - Returns the pyramid

4. **Creating Laplacian Pyramids:**
    - Generate Laplacian pyramids with the `laplacian_pyramid` function.
    - Returns the pyramid

5. **Reconstructing the Pyramid:**
    - Reconstruct the pyramid using the `reconstructing_pyramid` function.

6. **Blending Pyramids:**
    - Blend two images by applying the Laplacian pyramid blending technique with the `blend_pyramids` function.

7. **Displaying Pyramids:**
    - Visualize the pyramids using the `displaying_pyramid` function.

8. **Mask Calculations:**
    - Generate various masks, such as left-right, top-bottom, and diagonal masks, using the corresponding functions.

9. **Selecting a Mask Region:**
    - Choose a region for masking using `select_mask_region`.

10. **Fitting Image to Mask:**
    - Resize and pad an image to match the selected mask region with `fit_image_to_mask`.

11. **Image Blending Modes:**
    - Two blending modes are available:
        - `blend_image_mode1`: Blends a single image with a selected region of another image.
        - `blend_image_mode2`: Blends two separate images using a specified mask.

12. **Running the Example:**
    - Update the file paths for the original and other images in the script.
    - Set the desired pyramid level and create a mask using functions or read the mask as grayscale using Image.open().convert("L").
    - Execute the script to observe the image blending results.
    - If want to see pyramids please adjust the blend_image function parameter display=True

## Example:

```python
img_org = plt.imread('miyeon1_clipped.jpg')
img_other = plt.imread('miyeon2_clipped.jpg')
pyramid_level = 10
pyramid_mask = create_half_mask_for(img_org)
last_image = blend_image_mode2(img_org, img_other, pyramid_mask, level=pyramid_level)
```

## Output:

- The script outputs the blended image and visualizations of Gaussian and Laplacian pyramids at each level.

## Note:

- Ensure that the image files are present in the specified file paths.
- Adjust parameters such as `pyramid_level` and masks based on the specific requirements of your image blending scenario.
