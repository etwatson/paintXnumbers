import numpy as np
import sys
from skimage import io, segmentation, color, img_as_ubyte
from sklearn.cluster import KMeans

def paint_by_numbers(image_path, n_colors=20):
    # Load image
    image = io.imread(image_path)

    # Quantize colors using KMeans clustering
    image_array = np.reshape(image, (image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)
    quantized_image = kmeans.cluster_centers_[labels].reshape(image.shape).astype('uint8')

    # Segment the image
    seg_map = segmentation.slic(quantized_image, n_segments=n_colors, compactness=10, start_label=1)
    segmented_image = color.label2rgb(seg_map, quantized_image, kind='avg', bg_label=0)

    # Save the segmented image
    segmented_image_rescaled = img_as_ubyte(segmented_image)
    output_path = image_path.replace('.jpg', '_paint_by_numbers.jpg')
    io.imsave(output_path, segmented_image_rescaled)

    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py path_to_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = paint_by_numbers(image_path)
    print(f"Processed image saved at: {output_path}")
