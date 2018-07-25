# _*_ coding:utf-8 _*_
__author__ = 'T'

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ImgCompress():
    def __init__(self):
        self.r = 512

    def compress(self, file_source, file_dest):
        im = Image.open(file_source)
        im.rotate(45).convert('RGB').save('d:/test2.jpg')
        print(np.matrix(im.convert('1').getdata()))
        width, height = im.size
        data = np.matrix(im.convert('1').getdata()).reshape(width, height) / 255
        U, D, Vt = np.linalg.svd(data)
        D_compress = np.diag(D[:self.r])
        new_data = U[:, :self.r].dot(D_compress).dot(Vt[:self.r,:])
        new_im = Image.fromarray(new_data.astype(np.uint8))
        new_im = new_im.convert('RGB')
        """
        data_alt = np.array(new_im.getdata())
        print(data_alt.shape)
        for i in range(width):
            for j in range(height):
                new_im.putpixel((i, j), new_im.getpixel((i, j)) * im.getpixel((i, j)))
        """
        new_im.save(file_dest)

    def svd(self):
        A = np.random.randint(0, 5, (3, 4))
        print(A)

        U,D,Vt = np.linalg.svd(A)
        print(U)
        print(D)
        print(Vt)
        print(U[:,:3].dot(np.diag(D)).dot(Vt[:3,:]))

    def eig(self):
        np.random.seed(3)
        A = np.random.randint(1,110, (3,3))
        print(A)
        eig_val, eig_vec = np.linalg.eig(A)
        print(eig_val)
        print(eig_vec)
        a = np.diag(eig_val)
        print(a)
        print(np.linalg.inv(eig_vec))
        C = eig_vec.dot(a).dot(np.linalg.inv(eig_vec))
        print(C)

if __name__ == "__main__":
    file_path = 'D:/648870-20151031110532232-1120681583.jpg'
    file_path_dest = 'D:/test.jpg'
    ic = ImgCompress()
    ic.compress(file_path, file_path_dest)