# 數據集介紹

![image](https://github.com/11024244/mid/blob/main/jpg/01.png)

數據集來自**Kaggle**，品質很高，由知名醫院的專業人員嚴格審核標註，如圖所示數據有4種類別：

 •**CNV**：具有新生血管膜和相關視網膜下液的脈絡膜新血管形成
 
 •**DME**：糖尿病性黃斑水腫與視網膜增厚相關的視網膜內液
 
 •**DRUSEN**：早期AMD中存在多個玻璃疣
 
 •**NORMAL**：視網膜正常，沒有任何視網膜液或水腫
 
![image](https://github.com/11024244/mid/blob/main/jpg/02.png)

檔大小約為5GB，8萬多張圖像，分為訓練，測試，驗證三個資料夾，每個資料夾按照種類不同分成4個子資料夾，其次是具體圖像檔。

# 數據集下載

**掛載資料夾**：

```py
from google.colab import drive

drive.mount('/content/gdrive/')
```
按照提示進行驗證，結果如下：

![image](https://github.com/11024244/mid/blob/main/jpg/03.png)

**kaggle資料下載**：

創建**kaggle**帳戶並下載**kaggle.json**檔。 創建帳戶這裡就不介紹了，創建完帳戶後在“我的帳戶”-“API”中選擇“CREATE NEW API TOKEN”，然後下載**kaggle.json**檔。

**建立kaggle資料夾**：
```py
!mkdir -p ~/.kaggle
```
**將kaggle.json資料夾複製到指定資料夾**：
```py
!cp /content/gdrive/My\ Drive/kaggle.json ~/.kaggle/
```
**測試是否成功**：
```py
!kaggle competitions list
```
![image](https://github.com/11024244/mid/blob/main/jpg/04.png)

**下載資料集**：
```py
!kaggle datasets download -d paultimothymooney/kermany2018
```
![image](https://github.com/11024244/mid/blob/main/jpg/05.png)

**將文件解壓至google雲盤**：
```py
!unzip "/content/OCT2017.zip" -d "/content/gdrive/My Drive"
```


# 數據讀取

訓練，測試資料夾：
```py
import os

train_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'train', '**', '*.jpeg')
test_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'test', '**', '*.jpeg')
```
有人不知道這裡的“ ** ”什麼意思，我舉例說明吧：
```py
Example:
      If we had the following files on our filesystem:
        - /path/to/dir/a.txt
        - /path/to/dir/b.py
        - /path/to/dir/c.py
      If we pass "/path/to/dir/*.py" as the directory, the dataset would
      produce:
        - /path/to/dir/b.py
        - /path/to/dir/c.py
```
# 數據處理
```py
def input_fn(file_pattern, labels,
             image_size=(224,224),
             shuffle=False,
             batch_size=64, 
             num_epochs=None, 
             buffer_size=4096,
             prefetch_buffer_size=None):

    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(labels))
    num_classes = len(labels)

    def _map_func(filename):
        label = tf.string_split([filename], delimiter=os.sep).values[-2]
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
        # vgg16模型图像输入shape
        image = tf.image.resize_images(image, size=image_size)
        return (image, tf.one_hot(table.lookup(label), num_classes))
    
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    
    # tensorflow2.0以后tf.contrib模块就不再维护了
    if num_epochs is not None and shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
    elif shuffle:
        dataset = dataset.shuffle(buffer_size)
    elif num_epochs is not None:
        dataset = dataset.repeat(num_epochs)
    
    # map默认是序列的处理数据，取消序列可加快数据处理
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=_map_func,
                                      batch_size=batch_size,
                                      num_parallel_calls=os.cpu_count()))
    
    # prefetch数据预读取，合理利用CPU和GPU的空闲时间
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    
    return dataset
```
# 模型訓練
```py
import tensorflow as tf
import os

# 设置log显示等级
tf.logging.set_verbosity(tf.logging.INFO)

# 数据集标签
labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# include_top:不包含最后3个全连接层
keras_vgg16 = tf.keras.applications.VGG16(input_shape=(224,224,3),
                                          include_top=False)
output = keras_vgg16.output
output = tf.keras.layers.Flatten()(output)
predictions = tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)(output)

model = tf.keras.Model(inputs=keras_vgg16.input, outputs=predictions)

for layer in keras_vgg16.layers[:-4]:
    layer.trainable = False
    
optimizer = tf.train.AdamOptimizer()
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,
              metrics=['accuracy'])
              
est_config=tf.estimator.RunConfig(log_step_count_steps=10)
estimator = tf.keras.estimator.model_to_estimator(model,model_dir='/content/gdrive/My Drive/estlogs',config=est_config)
BATCH_SIZE = 32
EPOCHS = 2

estimator.train(input_fn=lambda:input_fn(test_folder,
                                         labels,
                                         shuffle=True,
                                         batch_size=BATCH_SIZE,
                                         buffer_size=2048,
                                         num_epochs=EPOCHS,
                                         prefetch_buffer_size=4))
```
