import random
import torchvision.transforms.functional as tf



class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        # self.PIL2Numpy = False

    def __call__(self, img1, mask):
        
        for a in self.augmentations:
            img1, mask = a(img1, mask)

        return img1, mask



class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img1,mask):
        if random.random() < self.p:
            return tf.hflip(img1),tf.hflip(mask)
        return img1,  mask


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img1, mask):
        if random.random() < self.p:
            return tf.vflip(img1),tf.vflip(mask)
        return img1, mask



class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img1, mask):
        # print(img1,img2,img3,mask)
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.rotate(img1, angle=rotate_degree
            ),  
            tf.rotate(mask, angle=rotate_degree
            ),
        )