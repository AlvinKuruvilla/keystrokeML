import os
import numpy as np
from image_similarity_measures.quality_metrics import fsim
from image_similarity import get_all_images_for_user, resize_to_smaller
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from rich.progress import track


def all_image_names():
    dir = "./result_images/"
    return [x.split(".")[0] for x in os.listdir(dir)]


def read_and_convert_image(image_path, use_root_dir: bool = True):
    if use_root_dir:
        img = Image.open("result_images/" + image_path + ".png")
    else:
        img = Image.open(image_path)
    img = np.asarray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img


def compute_fsim_pair(images, i, j):
    img1, img2 = resize_to_smaller(images[i], images[j])
    return i, j, fsim(img1, img2)


class HeatMap:
    def __init__(self) -> None:
        pass

    def create_from_all_images(self):
        if os.path.exists("matrix_data.dat"):
            return np.load("matrix_data.dat", allow_pickle=True)
        num_images = len(all_image_names())

        fsim_matrix = np.zeros((num_images, num_images))
        images = [read_and_convert_image(path) for path in all_image_names()]

        for i in track(range(num_images)):
            for j in range(
                i + 1, num_images
            ):  # Start j from i+1 to avoid duplicate calculations
                img1, img2 = resize_to_smaller(images[i], images[j])
                fsim_value = fsim(img1, img2)
                print(fsim_value)
                fsim_matrix[i, j] = fsim_value
                # Copying the FSIM value to the symmetric position
                fsim_matrix[j, i] = fsim_value

            fsim_matrix[i, i] = 1.0  # FSIM score of an image with itself is 1
        self.pickle_matrix(fsim_matrix)
        return fsim_matrix

    def create_from_user(self, user_id):
        images = [
            read_and_convert_image(path, use_root_dir=False)
            for path in get_all_images_for_user(user_id)
        ]
        num_images = len(images)
        fsim_matrix = np.zeros((num_images, num_images))
        for i in range(num_images):
            for j in range(
                i + 1, num_images
            ):  # Start j from i+1 to avoid duplicate calculations
                img1, img2 = resize_to_smaller(images[i], images[j])
                fsim_value = fsim(img1, img2)
                fsim_matrix[i, j] = fsim_value
                # Copying the FSIM value to the symmetric position
                fsim_matrix[j, i] = fsim_value

            fsim_matrix[i, i] = 1.0  # FSIM score of an image with itself is 1

        return fsim_matrix

    def plot(self, matrix, filename):
        plt.figure(figsize=(50, 45))  # Create a new figure with a specified size
        # Adjust vmin and vmax to ensure the color scale includes all data range
        # and choose a colormap that enhances visual differentiation
        ax = sns.heatmap(matrix, linewidth=0.5)
        ax.set_title(filename)  # Set the title
        plt.colorbar(ax.collections[0])  # Ensure color bar is added
        plt.savefig(filename)  # Save the figure
        plt.close()  # Close the figure to prevent color bar duplication

    def pickle_matrix(self, matrix):
        matrix.dump(
            "matrix_data.dat",
        )
