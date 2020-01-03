# TF Implementation of SATD

This is the implementation of Sum of Absolute Transformed Difference (SATD) in TensorFlow.
The implementation is used as the loss function in:

> "Progressive Spatial Recurrent Neural Network for Intra Prediction" <br />
> Yueyu Hu, Wenhan Yang, Mading Li, and Jiaying Liu <br />
> https://arxiv.org/abs/1807.02232

Numpy and TensorFlow are required.
The code can be used simply by calling the function:

```def SATD(y_true, y_pred, scale, batch_size=1, norm='L1')```

where ```y_true``` and ```y_pred``` stand for the ground truth signal and the predicted signal. Scale in ```{4,8,16,32}``` is supported. The code can be easily modified for other scales.