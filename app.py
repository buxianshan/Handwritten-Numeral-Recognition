import os
import glob
import tensorflow as tf
import mnist_inference
import mnist_train
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2


def evaluate(pic, pic_name):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x:[pic]}
        y = mnist_inference.inference(x, None)
        result = tf.argmax(y, 1)
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                number = sess.run(result, feed_dict=validate_feed)
                pic_name = pic_name.split('\\')[-1]
                print(pic_name,' is :',number[0])
            else:
                print('No checkpoint file found')
                return


def image_prepare(pic_name): 
    # 读取图像，第二个参数是读取方式
    img = cv2.imread(pic_name, 1)
    # 使用全局阈值，降噪
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # 把opencv图像转化为PIL图像
    im = Image.fromarray(cv2.cvtColor(th1,cv2.COLOR_BGR2RGB))
    # 灰度化
    im = im.convert('L')
    # 为图片重新指定尺寸
    im = im.resize((28,28), Image.ANTIALIAS)
    # plt.imshow(im)
    # plt.show()
    # 图像转换为list
    im_list = list(im.getdata())
    # 图像灰度反转
    result = [(255-x)*1.0/255.0 for x in im_list]
    return result


def main(argv=None):
    # 把要识别的图片放到下面的文件夹中
    img_path = 'picture/'
    imgs = glob.glob(os.path.join(img_path, '*'))
    for p in imgs:
        # 图像处理：降噪、灰度化、修改尺寸以及灰度反转
        pic = image_prepare(p)
        # 识别图像
        evaluate(pic, p)


if __name__ == '__main__':
    main()
