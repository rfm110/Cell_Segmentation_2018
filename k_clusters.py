import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage




test_array = [[0,0,0,0,1]]
#credirs : scikit-image
def find_binary_contours():
    x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
    r = np.sin(np.exp((np.sin(x) ** 3 + np.cos(y) ** 2)))

    contours = measure.find_contours(r, 1)
    fig, ax = plt.subplots()
    ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def find_binary_contours2():
    #use mathworks boundarymask
    a = np.zeros((50, 50))
    a[10:30, 10:30] = 1
    a[35:45, 35:45] = 2

    distance = ndimage.distance_transform_edt(a)

    distance[distance != 1] = 0
    plt.imshow(distance)
    plt.show()

    np.where(distance == 1)


# can use skimage.regionprops to find "Perimeter" in addition to "Area"


# credit : https://github.com/analyticalmonk/KMeans_elbow/blob/master/kmeans_elbow.py
def Kmeans_and_elbow(image, maxK=5, seed_centroids=None):
    """
        parameters:
        - data: pandas DataFrame (data to be fitted)
        - maxK (default = 10): integer (maximum number of clusters with which to run k-means)
        - seed_centroids (default = None ): float (initial value of centroids for k-means)
    """
    sse = {}
    for k in range(1, maxK):
        print("k: ", k)
        if seed_centroids is not None:
            seeds = seed_centroids.head(k)
            kmeans = KMeans(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(image)
            image["clusters"] = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(image)
            image["clusters"] = kmeans.labels_
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[k] = kmeans.inertia_
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()
    return




find_binary_contours2()






