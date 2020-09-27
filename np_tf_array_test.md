

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import math
```

### 生成随机数组


```python
data = np.random.randn(500,35,19)
print(data.shape)
```

    (500, 35, 19)
    

# 1. 矩阵的维度扩展
### 1.1 扩展一个空维度


```python
y = np.expand_dims(data, axis=-1)
print(y.shape)
```

    (500, 35, 19, 1)
    


```python
y = data[..., np.newaxis]
print(y.shape)
```

    (500, 35, 19, 1)
    


```python
y = data.reshape(500,35,19,1)
print(y.shape)
```

    (500, 35, 19, 1)
    

### 1.2 通过复制扩展张量


```python
y = tf.tile(data, [1, 2, 3]) # 后边的shape代表对应维度上的复制扩展倍数
print(data.shape)
print(y.shape)
```

    (500, 35, 19)
    (500, 70, 57)
    

### 1.3 通过滑动窗口进行取样扩展
#### 方法一：直接滑动窗口进行扩展_拼接列表


```python
def get_train_data_window(data_inp, window_size): # 此处使用的是滑动窗口法获取的数据
    '''
    Create data using sliding windows
    Args:
        df: input data
        window_len: window length
    Return:
        np.array(data): shape (loop, seq, ...)
        loop: less loops but all data is used.
    example:
        [[a] 
        [b]
        [c]
        [d]
        [e]]
        [[a b c]
        [b c d]
        [c d e]]
    '''
    data = []
    loop = len(data_inp) - window_size + 1
    for i in range(loop):
        window = data_inp[i : i + window_size] # (seq, ...)
        data.append(window)
    return np.array(data) # (loop, seq, ...)
```


```python
y = get_train_data_window(data, 18)
print(y.shape)
y1 = np.transpose(y, axes=(0,2,3,1))
print(y1[...,:12].shape) # train data
print(y1[...,12:].shape) # label to be squeezed
```

    (483, 18, 35, 19)
    (483, 35, 19, 12)
    (483, 35, 19, 6)
    

#### 方法二：扩展窗口大小的数据然后直接变换
* get_time_series_window: 复制window size的数据后并在前或后设置延迟
* mode: train or label
    * mode='train': 设置开始延迟，前边的数据用第一个值填充，用于产生训练输入数据
    * mode='label': 设置结束延迟，后续的数据用nan或者其他填充，用于产生训练label，之后使用filter将有nan标签的数据去除掉

* 写法：合并空维度


```python
def get_time_series_window(x, window_size, mode='train'): 
    '''
    example:
        [[a] 
        [b]
        [c]
        [d]]
        mode='train'
        [[a a a]
        [a a b]
        [a b c]
        [b c d]]
        mode='label'
        [[a b c]
        [b c d]
        [c d nan]
        [d nan nan]]
    '''
    x = np.expand_dims(x, axis=-1) # 扩展一个空维度，最后在这个空维度上合并
    init_value = x[0]
    inputs_lagged = x
    for i in range(1, window_size): # i(1,2,...,window_size-1) i=0时已经确定了
        if mode == 'train':
            inputs_roll = np.roll(x, i, axis=0) # 按照某一轴的数据顺序滚动i次
            inputs_roll[:i] = init_value
            inputs_lagged = np.concatenate((inputs_roll, inputs_lagged), axis=-1)
        elif mode == 'label':
            inputs_roll = np.roll(x, -i, axis=0) # 按照某一轴的数据逆序滚动i次
            inputs_roll[-i:] = np.nan
            inputs_lagged = np.concatenate((inputs_lagged, inputs_roll), axis=-1)
    return inputs_lagged
```


```python
x = get_time_series_window(data, 12, 'train')
print(x.shape)
y = get_time_series_window(data, 6, 'label')
print(y.shape)
```

    (500, 35, 19, 12)
    (500, 35, 19, 6)
    


```python
filter = (np.isnan(y)).any(axis=-1)[:,0,0] # findout which first axis nan data to remove 
print(filter.shape)
x = x[~filter]
y = y[~filter]
print('x:', x.shape,'y:', y.shape)
```

    (500,)
    x: (495, 35, 19, 12) y: (495, 35, 19, 6)
    

# 2. 矩阵的维度缩减
### 2.1 压缩所有空维度


```python
y = np.expand_dims(data, axis=-1)
print(y.shape)
y = np.squeeze(y) # tf.squeeze _ the same
print(y.shape)
```

    (500, 35, 19, 1)
    (500, 35, 19)
    

### 2.2 减少维度
* tf.reduce_sum, mean, max, min, prod, all: 按照对应的方法缩减维度


```python
y = tf.reduce_sum(data, axis=-1, keepdims=False)
print(y.shape)
```

    (500, 35)
    

# 3. 矩阵维度变换


* 通用维度转换——reshape函数

* 交换矩阵维度


```python
y1 = np.swapaxes(data, 1, 2) 
print(y1.shape)
```

    (500, 19, 35)
    

* 矩阵的转置(包括交换随便两个维度轴)
* 同理还可以用tf.transpose进行计算


```python
y = np.transpose(data)
print(y.shape)
y2 = np.transpose(data, axes=(0, 2, 1))
print(y2.shape)
print(np.array_equal(y1,y2))
```

    (19, 35, 500)
    (500, 19, 35)
    True
    


```python
!jupyter nbconvert --to markdown np_tf_array_test.ipynb
```


```python

```
