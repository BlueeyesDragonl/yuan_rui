import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
try:
  # %tensorflow_version 仅存在于Colab中
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

print("Tensorflow version " + tf.__version__)

# Matplotlib 配置
plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')
# Matplotlib 字体
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")

# 显示预测结果的工具函数
def display_images(digits, predictions, labels, title):
  n = 10
  indexes = np.random.choice(len(predictions), size=n)
  n_digits = digits[indexes]
  n_predictions = predictions[indexes]
  n_predictions = n_predictions.reshape((n,))
  n_labels = labels[indexes]
 
  fig = plt.figure(figsize=(20, 4))
  plt.title(title)
  plt.yticks([])
  plt.xticks([])

  for i in range(10):
    ax = fig.add_subplot(1, 10, i+1)
    class_index = n_predictions[i]
    plt.xlabel(classes[class_index])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(n_digits[i])

# 加载 CIFAR-10 数据集
(training_images, training_labels), (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
display_images(training_images, training_labels, training_labels, "Training Data")
display_images(validation_images, validation_labels, validation_labels, "Validation Data")

# 图像预处理
def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims
train_X = preprocess_image_input(training_images)
valid_X = preprocess_image_input(validation_images)

# 特征提取由预训练的 ResNet50 模型完成
def feature_extractor(inputs):
  feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
  return feature_extractor

# 定义最终的全连接层和 softmax 层进行分类
def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

# 由于输入图像尺寸为 (32 x 32)，首先通过上采样因子 (7x7) 将其转换为 (224 x 224)，
# 然后连接特征提取和 "分类器" 层以构建模型
def final_model(inputs):
    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)
    return classification_output

# 定义模型并进行编译
# 使用随机梯度下降（SGD）作为优化器
# 使用稀疏分类交叉熵作为损失函数
def define_compile_model():
  inputs = tf.keras.layers.Input(shape=(32,32,3))
  classification_output = final_model(inputs) 
  model = tf.keras.Model(inputs=inputs, outputs = classification_output)
  model.compile(optimizer='SGD', 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  return model

model = define_compile_model()
model.summary()

EPOCHS = 3
history = model.fit(train_X, training_labels, epochs=EPOCHS, validation_data=(valid_X, validation_labels), batch_size=64)
loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=64)

# 显示训练和验证曲线
def plot_loss(loss):
    plt.plot(history.history[loss], "-bx")
    plt.plot(history.history['val_' + loss],"-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train loss","val loss"])
plot_loss("loss")

def plot_acc(acc):
    plt.plot(history.history[acc], "-bx")
    plt.plot(history.history['val_' + acc], "-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train accuracy", "val accuracy"])
plot_acc("accuracy")

probabilities = model.predict(valid_X, batch_size=64)
probabilities = np.argmax(probabilities, axis=1)
display_images(validation_images, probabilities, validation_labels, "Bad predictions indicated in red.")

import pandas as pd
probabilities = model.predict(valid_X, batch_size=64)
predictions = np.argmax(probabilities, axis=1)
# 创建ID列
ids = np.arange(1, len(predictions)+1)

# 创建DataFrame
submission_df = pd.DataFrame({
    'Id': ids,
    'Category': predictions
})

# 保存为CSV文件
submission_df.to_csv('predictions.csv', index=False)