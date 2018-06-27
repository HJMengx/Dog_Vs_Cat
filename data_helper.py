import numpy as np
import threading
import keras
import cv2

# custom keras generator
class MXGenerator(keras.utils.Sequence):
    def __init__(self,data,n,des_size=(224,224),means=None,stds=None,is_directory=True,batch_size=32,shuffle=True,seed=0):
        '''
        data: tuple of (x,y)
        n: data size
        des_size: standard size
        means: the dataset mean of RGB,default is imagenet means [103.939, 116.779, 123.68]
        batch_size: default is 32
        shuffle: random the data,default is True
        '''
        self.x = np.array(data[0])
        if len(data) >= 2:
            self.y = np.array(data[1])
        else:
            self.y = None
        self.n = n
        self.des_size = des_size
        self.is_directory = is_directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lock = threading.Lock()
        self.index_array = self._set_index_array()
#         self.index_generator = self._flow_index()
        self.means = means
        self.stds = stds
        #super(MXGenerator,self).__init__()
        
    def reset_index(self):
        self.batch_index = 0
    
    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.index_array)
            
    def on_epoch_end(self):
        self._set_index_array()
    
    def __len__(self):
        # batch count
        return int(np.ceil(self.n / self.batch_size))
    # keras will call this function for data if the class is subclass of Sequence,otherwise will call the next
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

#          # if not use index_generator, this function is not call
#     def _flow_index(self):
#         # data generate
#         self.batch_size = 0
#         while True:
#             if self.seed is not None:
#                 np.random.seed(self.seed + self.total_batches_seen)
#             if self.batch_index == 0:
#                 self._set_index_array()

#             current_index = (self.batch_index * self.batch_size) % self.n
#             # batch_index will be set 0 when the value * batch_size large to the n
#             if self.n > current_index + self.batch_size:
#                 self.batch_index += 1
#             else:
#                 self.batch_index = 0
#             self.total_batches_seen += 1
            
#             yield self.index_array[current_index:
#                                    current_index + self.batch_size]
#     # in python3, __next__,2 is next
#     def __next__(self, *args, **kwargs):
#         return self.next()
    
#     def next(self):
#         with self.lock:
#             # 
#             index_array = next(self.index_generator)
#         return self._data_generate(index_array)   