import os
import numpy as np
import sys
from skimage import io, segmentation, color, img_as_ubyte
from sklearn.cluster import KMeans

def paintXnumbers(image_path, n_colors=20):
    # Load image
    image = io.imread(image_path)

    # Quantize colors using KMeans clustering
    image_array = np.reshape(image, (image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)
    quantized_image = kmeans.cluster_centers_[labels].reshape(image.shape).astype('uint8')

    # Segment the image
    seg_map = segmentation.slic(quantized_image, n_segments=n_colors, compactness=10, start_label=1)
    segmented_image = color.label2rgb(seg_map, quantized_image, kind='avg', bg_label=0)

    # Get the filename without the extension
    base_name = os.path.basename(image_path)
    file_name_without_ext = os.path.splitext(base_name)[0]

    # Convert the segmented_image to the proper data type for saving
    segmented_image_rescaled = img_as_ubyte(segmented_image)

    # Define the output path in the current directory
    output_filename = f"{file_name_without_ext}_paintX.jpg"
    output_path = os.path.join(os.getcwd(), output_filename)

    # Save the image
    io.imsave(output_path, segmented_image_rescaled)

    return output_path

if __name__ == "__main__":
    image_path = sys.argv[1]
    output_path = paintXnumbers(image_path)
    print(f"Processed image saved at: {output_path}")