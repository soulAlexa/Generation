import numpy

def open(path):
    data = numpy.load('face_images.npz')
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])


if __name__ == '__main__':
    open(123)
