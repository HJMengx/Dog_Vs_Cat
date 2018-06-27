### 猫狗大战

[项目地址](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

[导出的文件以及源码(密码:i39y)](https://pan.baidu.com/s/1VqZKqA44o1M_lSFYWQscXQ)

#### 读取数据

图片文件通过文件名来代表类别.例如:`dog.1.jpg,cat.1.jpg`.

```
def load_data(load_type="train"):
    path = None
    n = 25000
    if load_type=="train":
        imgs = []
        labels = []
        
        path = "train/"
        img_names = os.listdir(path)
        
        for name in img_names:
            imgs.append('train/'+name)
            labels.append([0] if name[:3] == 'cat' else
             [1])
            train_img_names,valid_img_names,train_labels,valid_labels = train_test_split( \
                                                        imgs, labels, test_size=0.2, random_state=42)
        return train_img_names,valid_img_names,train_labels,valid_labels
    else:
        # test,don`t have the labels
        path = 'test/'
        img_names = os.listdir(path)
        imgs = []
        for img in img_names:
            imgs.append(img)
                
        return imgs
```

### 自定义生成器

`keras`生成器从目录生成需要配置一下软连接,按类别分目录,所以我直接继承了`keras.utils.Sequence`类,核心是实现`__getitem__`方法,这是数据的产生.

```
    def __getitem__(self, idx):
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._data_generate(index_array)
    
    def _data_generate(self,index_array):
        # read from path
        # request the memory
        imgs = np.zeros((len(index_array),self.des_size[0],self.des_size[1],3),dtype=np.uint8)
        lables = None
        # read the data
        if self.is_directory:
            img_names = self.x[index_array]
            for name_index in range(len(img_names)): #range(0,((index+1)*self.batch_size - index*self.batch_size))
                img = cv2.imread(img_names[name_index])
                if img is not None:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img,self.des_size)
                    imgs[name_index] = img 
        else:
            for i in range(len(index_array)):
                img = self.x[index_array[i]]
                img = cv2.resize(img,self.des_size)
                imgs[i] = img
                
        if self.y is not None:
            labels = self.y[index_array]
        if labels is None:
            return imgs
        else:
           # print(img_names,labels)
            return imgs,labels
```

`__len__`是总共能有多少个`batch`.

`_set_index_array`重置`index_array`保证每一个`epoch`都能产生不同位置的`batch`,随机性够大,模型能具有更大的泛化性.

`on_epoch_end`在每一个`epoch`结束的时候,`keras`会调用传入`generator`的该方法.

#### 异常值筛选

`imagenet`里包含了猫狗类别,所以可以用在`imagenet`上训练过的模型来辨别是否含有猫狗. 比如top50是否包含,多结合几个模型共同判断效果更佳.

借用他人对`imagenet`的类别归总:

```
dogs = [
 'n02085620','n02085782','n02085936','n02086079'
,'n02086240','n02086646','n02086910','n02087046'
,'n02087394','n02088094','n02088238','n02088364'
,'n02088466','n02088632','n02089078','n02089867'
,'n02089973','n02090379','n02090622','n02090721'
,'n02091032','n02091134','n02091244','n02091467'
,'n02091635','n02091831','n02092002','n02092339'
,'n02093256','n02093428','n02093647','n02093754'
,'n02093859','n02093991','n02094114','n02094258'
,'n02094433','n02095314','n02095570','n02095889'
,'n02096051','n02096177','n02096294','n02096437'
,'n02096585','n02097047','n02097130','n02097209'
,'n02097298','n02097474','n02097658','n02098105'
,'n02098286','n02098413','n02099267','n02099429'
,'n02099601','n02099712','n02099849','n02100236'
,'n02100583','n02100735','n02100877','n02101006'
,'n02101388','n02101556','n02102040','n02102177'
,'n02102318','n02102480','n02102973','n02104029'
,'n02104365','n02105056','n02105162','n02105251'
,'n02105412','n02105505','n02105641','n02105855'
,'n02106030','n02106166','n02106382','n02106550'
,'n02106662','n02107142','n02107312','n02107574'
,'n02107683','n02107908','n02108000','n02108089'
,'n02108422','n02108551','n02108915','n02109047'
,'n02109525','n02109961','n02110063','n02110185'
,'n02110341','n02110627','n02110806','n02110958'
,'n02111129','n02111277','n02111500','n02111889'
,'n02112018','n02112137','n02112350','n02112706'
,'n02113023','n02113186','n02113624','n02113712'
,'n02113799','n02113978']

cats=[
'n02123045','n02123159','n02123394','n02123597'
,'n02124075','n02125311','n02127052']
```

辨别代码:

```
all_train_generator_224 = MXGenerator((all_train,all_laebls),len(all_train),des_size=(224,224),
               batch_size=batch_size,shuffle=False)

## 只操作一个,其他类似
predict_res = res_nn.predict_generator(all_train_generator_224, 
                                       steps=len(all_train)//batch_size, use_multiprocessing=True, verbose=1)

# decode_predictions
results = resnet50.decode_predictions(predict_res,top=50)

# 存储无异常的图片和标签
no_exception_img = []
no_exception_label = []

# 查看是否包含在dogs or cats
# n * 30
for result_index in range(len(results)):
    result = results[result_index]
    is_normal = False
    for classes in result:
        if classes[0] in dogs or classes[0] in cats:
            # normal
            is_normal = True
            break
    if is_normal:
        no_exception_true.append(all_train[result_index])
        no_exception_label.append(all_laebls[result_index])
    else:
        print("not cat or dog image:",all_train[result_index])                           
```

#### 迁移学习

在`imagenet`上训练过的模型,修改输出层.

```
## resnet
base_model = ResNet50(input_tensor=Lambda(resnet50.preprocess_input)(Input(shape=(224,224,3))), 
                      weights='imagenet', include_top=False)

for layers in base_model.layers:
    layers.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.25)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(base_model.input, x)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit_generator(train_generator_224,len(train_img_names)//batch_size,epochs=5,
                    validation_data=valid_generator_224,validation_steps=len(valid_img_names)//batch_size,shuffle=False)
```

#### 特征向量的导出

导出各个微调后的模型的特征向量,全局平均池化层后的输出.

`(n,2048)`.

```
import h5py

def write_feature(model_name,model,train_generator,train_labels,valid_generator,valid_labels,test_generator,batch_size=32):
    if model_name == 'resnet_feature':
        model.load_weights('resnet.h5',by_name=True)
    elif model_name == 'inception_feature':
        model.load_weights('incep.h5',by_name=True)
    else:
        model.load_weights('xcep.h5',by_name=True)
    # 转换为numpy数组
    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)
    
    train_feature = model.predict_generator(train_generator,int(np.ceil(train_generator.samples/batch_size)),verbose=1)
    valid_feature = model.predict_generator(valid_generator,int(np.ceil(valid_generator.samples/batch_size)),verbose=1)
    test_feature  = model.predict_generator(test_generator,int(np.ceil(test_generator.samples/batch_size)),verbose=1)
    print("train_feature.shape:",train_feature.shape)
    print("valid_feature.shape:",valid_feature.shape)
    with h5py.File(model_name+'.h5','w') as file:
        file.create_dataset("train",data=train_feature,dtype="float32")
        file.create_dataset('trian_labels',data=np.array(train_generator.classes),dtype="uint8")
        file.create_dataset("valid",data=valid_feature,dtype="float32")
        file.create_dataset("valid_labels",data=np.array(valid_generator.classes),dtype="uint8")
        file.create_dataset("test",data=test_feature,dtype="float32")       
```

```
# resnet50
write_feature('resnet_feature',Model(inputs=model.input,outputs=model.layers[-3].output),
              train_generator_224,train_labels,valid_generator_224,valid_labels,test_generator_224)
```

#### 模型融合

结合导出的特征,在融合特征上训练一个分类模型.

```
feature_files = ['resnet_feature.h5','inception_feature.h5','xception_feature.h5']

X_train = []
y_train = []
X_valid = []
y_valid = []
X_test = []

for file_name in feature_files:
    with h5py.File(file_name, 'r') as h:
        X_train.append(np.array(h['train']))
        X_valid.append(np.array(h['valid']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['trian_labels'])
        y_valid = np.array(h['valid_labels'])
        print(np.array(h['train']).shape,np.array(h['valid']).shape,np.array(h['test']).shape)
# concatenate
# print(X_train.shape,X_valid.shape,X_test.shape,y_train.shape,y_valid.shape)

X_train = np.concatenate(X_train, axis=1)
X_valid = np.concatenate(X_valid, axis=1)
X_test = np.concatenate(X_test, axis=1)
```

##### `Train`

```
input_tensor = Input(X_train.shape[1:])
x = input_tensor
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)
concatenate_model = Model(inputs=input_tensor, outputs=x)

concatenate_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
import keras.utils

# keras.utils.plot_model(concatenate_model,to_file='model.png')

concatenate_model.fit(X_train,y_train,batch_size=128, epochs=5,validation_data=(X_valid,y_valid))
```

##### `result`

```
Train on 20000 samples, validate on 5000 samples
Epoch 1/5
20000/20000 [==============================] - 6s 278us/step - loss: 0.0569 - acc: 0.9796 - val_loss: 0.0169 - val_acc: 0.9940
Epoch 2/5
20000/20000 [==============================] - 1s 45us/step - loss: 0.0198 - acc: 0.9937 - val_loss: 0.0152 - val_acc: 0.9938
Epoch 3/5
20000/20000 [==============================] - 1s 45us/step - loss: 0.0166 - acc: 0.9945 - val_loss: 0.0139 - val_acc: 0.9940
Epoch 4/5
20000/20000 [==============================] - 1s 45us/step - loss: 0.0142 - acc: 0.9958 - val_loss: 0.0131 - val_acc: 0.9950
Epoch 5/5
20000/20000 [==============================] - 1s 49us/step - loss: 0.0130 - acc: 0.9959 - val_loss: 0.0126 - val_acc: 0.9946
```

#### 生成预测

用新的模型在测试集进行预测.

```
import pandas as pd
y_pred = concatenate_model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)

df = pd.read_csv("pred.csv")

image_size = (224, 224)
gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("test2", image_size, shuffle=False, 
                                         batch_size=16, class_mode=None)

for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', y_pred[i])

df.to_csv('predict.csv', index=None)

print(df)
```

最后能得到`kaggle`上`0.0386`的`loss`.

### 在新图上进行预测

新的图片进来的时候,首先需要通过三个模型导出特征向量,然后在用融合模型进行预测.

```
def predict(input_image):
    if input_image is None:
        return 
    if type(input_image) != type(np.array()):
        return 
    # resnet model
    res = Model(inputs=model.input,outputs=model.layers[-3].output)
    inception = Model(inputs=inception_model.input,outputs=inception_model.layers[-3].output)
    xcep = Model(inputs=xcep_model.input,outputs=xcep_model.layers[-3].output)
    
    res_feature = res.predict(np.expand_dims(cv2.resize(input_image,(224,224)),axis=0))
    incep_feature = inception.predict(np.expand_dims(cv2.resize(input_image,(299,299)),axis=0))
    xcep_feature = xcep.predict(np.expand_dims(cv2.resize(input_image,(299,299)),axis=0))
```