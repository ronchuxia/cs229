from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


def main():
    k = 16

    image_small = imread('../data/peppers-small.tiff')
    plt.imshow(image_small)
    plt.savefig('output/image_small.png')

    image_large = imread('../data/peppers-large.tiff')
    plt.imshow(image_large)
    plt.savefig('output/image_large.png')

    h_small, w_small, c_small = image_small.shape
    image_small = np.reshape(image_small, (h_small * w_small, c_small))

    # initialize centroids
    num_pixels = h_small * w_small
    centroids_idx = np.random.choice(range(num_pixels), k)
    centroids = image_small[centroids_idx]  # shape (k, c_small)

    # run k-means algorithm
    centroids = kmeans(image_small, centroids)

    # classify the pixels
    h_large, w_large, c_large = image_large.shape
    image_large = np.reshape(image_large, (h_large * w_large, c_large))
    distances = np.linalg.norm(np.expand_dims(image_large, axis=1) - np.expand_dims(centroids, axis=0), axis=2)   # shape (h_large * w_large, k)
    labels = np.argmin(distances, axis=1)   # shape (h_large * w_large, k)
    
    image_large_compressed = centroids[labels]
    image_large_compressed /= 256
    image_large_compressed = np.reshape(image_large_compressed, (h_large, w_large, c_large))
    plt.imshow(image_large_compressed)
    plt.savefig('output/image_large_compressed.png')


def kmeans(x, centroids):
    """k-means algorithm
    
    Args: 
        x: shape (m, n)
        centroids: shape (k, n)
        k: number of clusters

    Returns:
        centroids: shape (k, n)
    """
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    m, n = x.shape
    k, _ = centroids.shape

    it = 0
    j = prev_j = None
    while it < max_iter and (prev_j is None or np.abs(j - prev_j) >= eps):
        distances = np.linalg.norm(np.expand_dims(x, axis=1) - np.expand_dims(centroids, axis=0), axis=2)   # shape (m, k)
        labels = np.argmin(distances, axis=1)   # shape (m, )
        indicator = np.zeros((m, k))
        indicator[range(m), labels] = 1 # shape (m, k)

        centroids = np.sum(np.expand_dims(indicator, axis=2) * np.expand_dims(x, axis=1), axis=0) / np.expand_dims(np.sum(indicator, axis=0), axis=1)   # shape (k, n)
        
        prev_j = j
        j = np.sum(np.linalg.norm(x - centroids[labels], axis=1))
        # print(j)

        it += 1

    print(it)

    return centroids


if __name__ == '__main__':
    main()