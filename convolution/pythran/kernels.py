import numpy as np
"""
Definit une convolution faisant une moyenne des voisins d'un pixel donne
( stencil de 3x3 )
"""


def convolve_mean(image):
    return 0.25*(image[:-2, 1:-1] + image[2:, 1:-1] +
                 image[1:-1, :-2] + image[1:-1, 2:])


def convolve_laplacien(image):
    """
    Definie l'operateur laplacien comme convolution :
        permet de detecter les bords dans une image
    """
    out_image = np.abs(4*image[1:-1, 1:-1] - image[:-2, 1:-1] -
                       image[2:, 1:-1] - image[1:-1, :-2] - image[1:-1, 2:])
    # On renormalise l'image :
    valmax = np.max(out_image)
    valmax = max(1., valmax)+1.E-9
    out_image *= 1./valmax
    return out_image


def convolve_matrix(image, convolution_array):
    """
    Convolution generale avec une taille de stencil quelconque.
    Permet de definir tous les stencils que l'on souhaite !
    """
    height = image.shape[0]
    width = image.shape[1]
    nx = convolution_array.shape[0]
    ny = convolution_array.shape[1]
    h = height - nx + 1
    w = width - ny + 1

    out_image = np.zeros_like(image[:h, :w])

    for jw in range(0, ny):
        for iw in range(0, nx):
            out_image += convolution_array[jw, iw] * image[jw:jw+h, iw:iw+w]

    # On renormalise l'image en ramenant les valeurs des couleurs entre 0 et 1
    out_image = np.abs(out_image)
    valmax = np.max(out_image)
    valmax = max(1., valmax) + 1.E-9
    out_image *= 1./valmax
    return out_image
