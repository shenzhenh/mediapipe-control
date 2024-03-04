import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
# 读取数据集 以及模型保存的位置
dataset = 'keypoint.csv'
model_save_path = 'gesture_classifier.hdf5'
tflite_save_path = 'gesture_classifier.tflite'

NUM_CLASSES = 9

# usecols指定读取的列数
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=0)
# 分割数据集  得到训练集以及测试集
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# 建立一个简单的深度学习模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2,)),
    # dropout层
    tf.keras.layers.Dropout(0.2),
    # 全连接层
    tf.keras.layers.Dense(20, activation='relu'),
    # dropout层
    tf.keras.layers.Dropout(0.4),
    # 全连接层
    tf.keras.layers.Dense(10, activation='relu'),
    # 输出层
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()
# 模型权重的保存方式以及位置
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
# 提前结束模型
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
# 参数配置
# 优化器为adam 损失函数为交叉损失熵 输出训练集和测试集的精确度
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# 开始模型的训练
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)
# 这里进行模型的验证
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)

# 导入模型文件
model = tf.keras.models.load_model(model_save_path)
# 进行预测
predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

# 输出更类型的预测结果 包括精确度等等
Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print(classification_report(y_test, y_pred))

# 保存一个专门用于后续推理的模型
model.save(model_save_path, include_optimizer=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
open(tflite_save_path, 'wb').write(tflite_quantized_model)

# 下面是对tflite格式的文件的测试
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))

# 进行预测
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))
