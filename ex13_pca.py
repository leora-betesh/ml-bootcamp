import shutil, os, warnings
import numpy as np
from scipy.misc import imread, toimage
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

root_dir = r'C:/LeoraProjects/bootcamp/ml-bootcamp/faces94/faces94/'
        
def convert_images_to_vectors():
    pixel_matrix = []
    for src_dir, dirs, files in os.walk(root_dir):
        if files:
            if files[0].endswith(".jpg") or files[0].endswith(".gif"):
                src_file = os.path.join(src_dir, files[0])
                img = imread(src_file).flatten().tolist()
                pixel_matrix.append(img)
    return pixel_matrix


def main():
    X = np.asarray(convert_images_to_vectors())
    mean_face =  np.asarray(np.mean(X,axis=0)).reshape((200,180,3))
    pca = PCA(n_components=50,svd_solver='randomized', whiten=True).fit(X)
    eigen_vectors = pca.components_
    weights = np.dot(eigen_vectors,X.T)

    #Choose face #1 to reconstruct
    reconstructed_face = mean_face
    for i in range(50):
        reconstructed_face += np.dot(weights[i,0],eigen_vectors.reshape((50, 200, 180, 3))[i])

    toimage(reconstructed_face).show()
    
if __name__ == '__main__':
   main()