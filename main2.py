import random

import PIL
from PIL import Image



def L_R(image):
    r = 0
    l = image.size[0]
    for y in range(0, image.size[0]):
        for x in range(0, image.size[1]):
            if image.getpixel((x, y)) == 0:
                if x < l:
                    l = x
                break
        for x in reversed(range(0, image.size[1])):
            if image.getpixel((x, y)) == 0:
                if x > r:
                    r = x
                break
    return l, r + 1

def Get_Pict_vector(image):
    size = image.size[0]
    vector = ''
    for y in range(0, image.size[0]):
        for x in range(0, image.size[1]):
            if image.getpixel((x, y)) != 0:
                vector += str(1)
            else:
                vector += str(0)
    return vector

def GEt_YOLO_R(image):
    up = image.size[1]
    down = 0
    r = 0
    l = image.size[0]
    for y in range(0, image.size[0]):
        for x in range(0, image.size[1]):
            if image.getpixel((x, y)) == 0:
                if x < l:
                    l = x
                if up > y:
                    up = y
                if down < y:
                    down = y
                break
        for x in reversed(range(0, image.size[1])):
            if image.getpixel((x, y)) == 0:
                if x > r:
                    r = x
                break
    down += 1
    r += 1
    center_X = (r - l)/2 + l
    center_y = (down - up)/2 + up
    return center_X/image.size[0], center_y/image.size[1], (r - l)/image.size[0], (down - up)/image.size[1]
def uniti(imgs):
    m1 = []
    sum = 0
    for i in imgs:
        l, r = L_R(i)
        sum += r - l
        ii = i.crop((l, 0, r, i.size[1]))
        m1.append(ii)
    Out = Image.new('L', (sum, sum), color=255)
    p = 0
    p1 = sum/2 - m1[0].size[1]/2
    p1 = round(p1)
    for i in range(0, len(m1)):
        # l, r = L_R(imgs[i])
        Out.paste(m1[i], (p, p1))
        p += m1[i].size[0]
    return Out

def generate(str, num):
    for n in range(num):
        imgs = []
        for i in str:
            t = Image.open(f'{i}image/test{random.randint(0, 100)}({i}).png')
            imgs.append(t)
        im = uniti(imgs)
        im = im.resize((64, 64))
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                if im.getpixel((x, y)) > 100:
                    im.putpixel((x, y), 255)
                else:
                    im.putpixel((x, y), 0)
        im.save(f'{str}image/test{n}.png')


if __name__ == '__main__':
    str1 = 'ABCEIKMOVY'
    num = 0
    fill = open('YoloDataset', 'w')
    for i in range(0, 10):
        fill.write(f'class {i} = {str1[i]} ')
    fill.write('\n')
    for i in str1:
        for ii in range(1, 51):
            im = Image.open(f'{i}image/test{ii}({i}).png')
            fill.write(f'{num}' + str(GEt_YOLO_R(im)) + Get_Pict_vector(im) + '\n')
        num += 1
    # for i in range(0, 100):
    # im = Image.open(f'Eimage/test3(E).png')
    # print(Get_Pict_vector(im))
    #     im1 = Image.open(f'Timage/test{}(T).png')
    #     im2 = Image.open(f'Cimage/test{}(C).png')
    #
    # im3 = uniti([im, im1, im2])
    #
    # # l, r = L_R(im2)
    # print(GEt_R(im))
    # # im3 = im2.crop((l, 0, r, im2.size[1]))
    # im3.save('testhtc.png')
    # generate('VIVO', 1000)

    # str1 = ['SONY', 'ZTE', 'VIVO', 'POCO', 'OPPO', 'NOKIA', 'MEIZU', 'HTC', 'HONOR', 'BLU']
    # num = 0
    # fill = open('YoloDataset2', 'w')
    # for i in range(0, 10):
    #     fill.write(f'class {i} = {str1[i]} ')
    # fill.write('\n')
    # for i in str1:
    #     for ii in range(0, 100):
    #         im = Image.open(f'{i}image/test{ii}.png')
    #         fill.write(f'{num}' + str(GEt_YOLO_R(im)) + '\n')
    #     num += 1