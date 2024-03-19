import tensorflow as tf

# 待转换文件
image_path = './input/x/1.jpg'

# 转换
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=1)  # 解码JPEG图像到灰度
image = tf.image.convert_image_dtype(image, tf.float32)  # 转换为[0,1]范围的浮点数
image = tf.image.resize(image, [28, 28])  # 调整图像大小

# 将浮点型张量转换为整型张量
image = tf.cast(image * 255, tf.uint8)

encoded_image = tf.image.encode_jpeg(image)

# 保存
tf.io.write_file('./train_x_1.jpg', encoded_image)
