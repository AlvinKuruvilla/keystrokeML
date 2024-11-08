import os
import re
import itertools
import cv2
import numpy as np
from rich.progress import track
from skimage.metrics import structural_similarity
from image_similarity_measures.quality_metrics import fsim
from scipy.spatial import distance


def get_all_images_for_user(user_id):
    files = []
    pattern = re.compile(
        rf"^{user_id}_" + r".*"
    )  # RegEx to match 'user_id_' at the start of the filename
    for filename in os.listdir("result_images"):
        if pattern.match(filename):
            files.append("result_images/" + filename)
    print(files)
    print(len(files))
    return files


def all_filename_pairs(filenames):
    return list(itertools.combinations(filenames, 2))


def structural_sim(img1, img2):
    sim, diff = structural_similarity(
        img1, img2, full=True, data_range=255, channel_axis=2
    )
    return sim


def resize_to_smaller(arr1, arr2):
    # Ensure both arrays are 3D by checking the number of dimensions and adjusting if necessary
    if arr1.ndim == 2:
        arr1 = arr1[:, :, np.newaxis]
    if arr2.ndim == 2:
        arr2 = arr2[:, :, np.newaxis]

    # Get dimensions of both arrays (excluding the channel dimension)
    rows1, cols1, _ = arr1.shape
    rows2, cols2, _ = arr2.shape

    # Determine which is smaller and resize the larger one
    if (rows1, cols1) > (rows2, cols2):
        arr1 = cv2.resize(arr1, (cols2, rows2), interpolation=cv2.INTER_AREA)
    elif (rows2, cols2) > (rows1, cols1):
        arr2 = cv2.resize(arr2, (cols1, rows1), interpolation=cv2.INTER_AREA)

    return arr1, arr2


def extract_color_histogram(image_path, bins=(8, 8, 8)):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Calculate histogram
    histogram = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

    # Normalize the histogram
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram


def mahalanobis_distance(hist1, hist2, covariance):
    inv_covariance = np.linalg.pinv(covariance)
    return distance.mahalanobis(hist1, hist2, inv_covariance)


def rmse(image1, image2):
    return np.sqrt(((image1 - image2) ** 2).mean())


def rmse_test():
    for i in track(range(1, 26)):
        if i == 22:
            continue
        pairs = all_filename_pairs(get_all_images_for_user(i))
        for pair in pairs:
            print(pair)
            image1_path, image2_path = pair
            img1 = cv2.imread(image1_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(image2_path)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img1, img2 = resize_to_smaller(img1, img2)
            # print("POST")
            # print(img1.shape)
            # print(img2.shape)
            print("RMSE:", rmse(img1, img2))


def fsim_test():
    for i in track(range(1, 26)):
        if i == 22:
            continue
        pairs = all_filename_pairs(get_all_images_for_user(i))
        for pair in pairs:
            print(pair)
            image1_path, image2_path = pair
            img1 = cv2.imread(image1_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(image2_path)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img1, img2 = resize_to_smaller(img1, img2)
            # Add a third dimension to grayscale images if needed
            if img1.ndim == 2:
                img1 = img1[:, :, np.newaxis]
            if img2.ndim == 2:
                img2 = img2[:, :, np.newaxis]
            print("FSIM:", fsim(img1, img2))


def structural_similarity_test():
    for i in track(range(1, 26)):
        if i == 22:
            continue
        pairs = all_filename_pairs(get_all_images_for_user(i))
        for pair in pairs:
            print(pair)
            image1_path, image2_path = pair
            img1 = cv2.imread(image1_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(image2_path)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img1, img2 = resize_to_smaller(img1, img2)
            print("Structural Similarity:", structural_sim(img1, img2))


def mahalanobis_test():
    for i in track(range(1, 26)):
        if i == 22:
            continue
        pairs = all_filename_pairs(get_all_images_for_user(i))
        for pair in pairs:
            print(pair)
            image1_path, image2_path = pair
            hist1 = extract_color_histogram(image1_path)
            hist2 = extract_color_histogram(image2_path)
            histograms = [hist1, hist2]  # Assume more histograms for a real scenario

            # Calculate covariance
            covariance = np.cov(np.stack(histograms, axis=0), rowvar=False)

            # Calculate Mahalanobis Distance
            dist = mahalanobis_distance(hist1, hist2, covariance)
            print("Mahalanobis Distance:", dist)


# fsim_test()
