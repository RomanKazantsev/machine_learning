/*
Copyright (c) 2018 Roman Kazantsev
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

import numpy as np
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # This function implementation is taken from:
    # https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    #assert (H + 2 * padding - field_height) % stride == 0
    #assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride) + 1
    out_width = int((W + 2 * padding - field_width) / stride) + 1
        
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    # This function implementation is taken from:
    # https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    # This function implementation is taken from:
    # https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Layer:
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:           output = layer.forward(input)

    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)

    Some layers also have learnable parameters which they update during layer.backward.
    """

    def __init__(self):
        """
        Here you can initialize layer parameters (if any) and auxiliary stuff.
        """
        return

    def forward(self, input):
        """
        Takes input data of shape [batch, ...], returns output data [batch, ...]
        """
        return

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input. Updates layer parameters and returns gradient for next layer
        Let x be layer weights, output – output of the layer on the given input and grad_output – gradient of layer with respect to output

        To compute loss gradients w.r.t parameters, you need to apply chain rule (backprop):
        (d loss / d x)  = (d loss / d output) * (d output / d x)
        Luckily, you already receive (d loss / d output) as grad_output, so you only need to multiply it by (d output / d x)
        If your layer has parameters (e.g. dense layer), you need to update them here using d loss / d x. The resulting update is a sum of updates in a batch.
        
        returns (d loss / d input) = (d loss / d output) * (d output / d input)
        """
        return

class ReLU(Layer):
    def __init__(self):
        """
        ReLU layer simply applies elementwise rectified linear unit to all inputs
        This layer does not have any parameters.
        """
        return

    def forward(self, input):
        """
        Perform ReLU transformation
        input shape: [batch, input_units]
        output shape: [batch, input_units]
        """
        zero_input = np.zeros(input.shape)
        result_input = np.maximum(input, zero_input)
        return result_input

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. ReLU input
        """
        doutput_by_dinput = np.zeros(input.shape)
        doutput_by_dinput[input > 0] = 1
        grad_input = np.multiply(doutput_by_dinput, grad_output)
        return grad_input

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = Wx + b

        W: matrix of shape [num_inputs, num_outputs]
        b: vector of shape [num_outputs]
        """
        variance = 2 / (input_units + output_units)
        self.learning_rate = learning_rate

        # initialize weights with small random numbers from normal distribution
        self.input_units = input_units
        self.output_units = output_units
        self.weights = np.random.normal(0, variance, (input_units, output_units))
        self.biases = np.random.normal(0, variance, output_units)
        return

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        fx = np.add(np.tensordot(input, self.weights, (1, 0)), self.biases)
        return fx

    def backward(self, input, grad_output):
        """
        input shape: [batch, input_units]
        grad_output: [batch, output units]

        Returns: grad_input, gradient of output w.r.t input
        """
        batch_size = input.shape[0]
        weights_exp = np.repeat(self.weights[np.newaxis, :, :], batch_size, axis=0)
        grad_output_exp = np.repeat(grad_output[:, np.newaxis, :], self.input_units, axis=1)
        intput_exp = np.repeat(input[:, :, np.newaxis], self.output_units, axis=2)

        grad_input = np.sum(np.multiply(weights_exp, grad_output_exp), axis=2)

        update = np.sum(np.multiply(intput_exp, grad_output_exp), axis=0)
        self.weights = self.weights + self.learning_rate * update
        return grad_input

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, learning_rate=0.1):
        """
        A convolutional layer with out_channels kernels of kernel_size.

        in_channels — number of input channels
        out_channels — number of convolutional filters
        kernel_size — tuple of two numbers: k_1 and k_2

        Initialize required weights.
        """
        kernel_size_x, kernel_size_y = kernel_size
        input_units = in_channels * kernel_size_x * kernel_size_y
        output_units = out_channels * kernel_size_x * kernel_size_y
        variance = 2 / (input_units + output_units)

        self.weights = np.random.normal(0, variance, (out_channels, in_channels, kernel_size_x, kernel_size_y))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size_x
        self.kernel_size_x = kernel_size_x
        self.kernel_size_y = kernel_size_y
        self.stride = 1
        self.stride_x = 1
        self.stride_y = 1
        self.input_col = np.zeros(1)
        self.learning_rate = learning_rate
        # array of shape [in_channels, out_channels, kernel_size, kernel_size]
        return
    
    def forward(self, input):
        """
        Perform convolutional transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """
        batch_size, in_channels, h, w = input.shape

        #if (h - self.kernel_size_y) % self.stride_y != 0 or (w - self.kernel_size_x) % self.stride_x != 0:
        #    raise Exception("Invalid input parameters. kernel_size_y = ", self.kernel_size_y, "kernel_size_x = ", self.kernel_size_x,
        #                    "w = ", w, "h = ", h, "self.stride_x = ", self.stride_x, "self.stride_y = ", self.stride_y)

        h_out = int((h - self.kernel_size_y) / self.stride_y + 1)
        w_out = int((w - self.kernel_size_x) / self.stride_x + 1)
        
        self.input_col = im2col_indices(input, self.kernel_size_y, self.kernel_size_x, padding=0, stride=self.stride)
        weights_col = self.weights.reshape(self.out_channels, -1)

        output = np.dot(weights_col, self.input_col)
        output = output.reshape(self.out_channels, h_out, w_out, batch_size)
        output = output.transpose(3, 0, 1, 2)
        return output

    def forward_new(self, input):
        """
        Perform convolutional transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """
        N, C, H, W = input.shape
        F, _, HH, WW = self.weights.shape
        stride, pad = self.stride, 0

        # Pad the input
        p = pad
        x_padded = np.pad(input, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        # Figure out output dimensions
        H += 2 * pad
        W += 2 * pad
        out_h = (H - HH) // stride + 1
        out_w = (W - WW) // stride + 1

        # Perform an im2col operation by picking clever strides
        shape = (C, HH, WW, N, out_h, out_w)
        strides = (H * W, W, 1, C * H * W, stride * W, stride)
        strides = input.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(x_padded,
                        shape=shape, strides=strides)
        self.input_col = np.ascontiguousarray(x_stride)
        self.input_col.shape = (C * HH * WW, N * out_h * out_w)

        # Now all our convolutions are a big matrix multiply
        res = self.weights.reshape(F, -1).dot(self.input_col)

        # Reshape the output
        res.shape = (F, N, out_h, out_w)
        out = res.transpose(1, 0, 2, 3)

        # Be nice and return a contiguous array
        # The old version of conv_forward_fast doesn't do this, so for a fair
        # comparison we won't either
        out = np.ascontiguousarray(out)

        return out

    def backward(self, input, grad_output):
        """
        Compute gradients w.r.t input and weights and update weights
        """
        grad_output_reshaped = grad_output.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        update_weights = np.dot(grad_output_reshaped, self.input_col.T)
        update_weights = update_weights.reshape(self.weights.shape)

        weights_reshape = self.weights.reshape(self.out_channels, -1)
        grad_input_col = np.dot(weights_reshape.T, grad_output_reshaped)
        grad_input = col2im_indices(grad_input_col, input.shape, self.kernel_size_y, self.kernel_size_x, padding=0, stride=self.stride)
        self.weights = np.add(self.weights, np.multiply(update_weights, self.learning_rate))
        return grad_input


class Maxpool2d(Layer):
    def __init__(self, kernel_size):
        """
        A maxpooling layer with kernel of kernel_size.
        This layer donwsamples [kernel_size, kernel_size] to
        1 number which represents maximum.

        Stride description is identical to the convolution
        layer. But default value we use is kernel_size to
        reduce dim by kernel_size times.

        This layer does not have any learnable parameters.
        """

        self.stride = kernel_size
        self.kernel_size = kernel_size
        self.max_ind = np.zeros(1)
        self.input_col = np.zeros(1)
        return

    def forward(self, input):
        """
        Perform maxpooling transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """
        batch_size, in_channels, h, w = input.shape

        if (h - self.kernel_size) % self.stride != 0 or (w - self.kernel_size) % self.stride != 0:
            raise Exception("Invalid parameters")
        
        h_out = int((h - self.kernel_size) / self.stride + 1)
        w_out = int((w - self.kernel_size) / self.stride + 1)

        input_reshaped = input.reshape(batch_size*in_channels, 1, h, w)

        self.input_col = im2col_indices(input_reshaped, self.kernel_size, self.kernel_size, padding=0, stride=self.stride)
        
        self.max_ind = np.argmax(self.input_col, axis=0)
        out = self.input_col[self.max_ind, range(self.max_ind.size)]

        out = out.reshape(h_out, w_out, batch_size, in_channels).transpose(2,3,0,1)
        return out

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. Maxpool2d input
        """
        batch_size, in_channels, h, w = input.shape
        grad_input_col = np.zeros(self.input_col.shape)
        grad_output_flat = grad_output.transpose(2,3,0,1).ravel()
        
        grad_input_col[self.max_ind,range(self.max_ind.size)] = grad_output_flat
        
        grad_input_shape = (batch_size*in_channels, 1, h, w)
        grad_input = col2im_indices(grad_input_col, (batch_size * in_channels, 1, h, w), self.kernel_size,
                                    self.kernel_size, padding=0, stride=self.stride)
        grad_input = grad_input.reshape(batch_size, in_channels, h, w)
        return grad_input

class Flatten(Layer):
    def __init__(self):
        """
        This layer does not have any parameters
        """
        return

    def forward(self, input):
        """
        input shape: [batch_size, channels, feature_nums_h, feature_nums_w]
        output shape: [batch_size, channels * feature_nums_h * feature_nums_w]
        """
        batch_size = input.shape[0]
        channels = input.shape[1]
        feature_nums_h = input.shape[2]
        feature_nums_w = input.shape[3]
        
        output = np.reshape(input, (batch_size, channels*feature_nums_h*feature_nums_w))
        return output

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. Flatten input
        """
        batch_size = input.shape[0]
        channels = input.shape[1]
        feature_nums_h = input.shape[2]
        feature_nums_w = input.shape[3]

        grad_input = np.reshape(grad_output,(batch_size, channels, feature_nums_h, feature_nums_w))
        return grad_input


def softmax_crossentropy_with_logits(logits, y_true):
    """
    Compute crossentropy from logits and ids of correct answers
    logits shape: [batch_size, num_classes]
    reference_answers: [batch_size]
    output is a number
    """
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=1)
    softmax = np.divide(exp_logits, np.repeat(sum_exp_logits[:, np.newaxis], num_classes, axis=1))
    y_true_bool = np.zeros([batch_size, num_classes])
    y_true_bool[np.arange(batch_size), y_true] = 1
    softmax_crossentropy = np.sum(np.multiply(y_true_bool, np.log(softmax))) / batch_size
    return softmax_crossentropy


def grad_softmax_crossentropy_with_logits(logits, y_true):
    """
    Compute crossentropy gradient from logits and ids of correct answers
    Output should be divided by batch_size, so any layer update can be simply computed as sum of object updates.
    logits shape: [batch_size, num_classes]
    reference_answers: [batch_size]
    """
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=1)
    softmax = np.divide(exp_logits, np.repeat(sum_exp_logits[:, np.newaxis], num_classes, axis=1))
    
    y_true_bool = np.zeros([batch_size, num_classes])
    y_true_bool[np.arange(batch_size), y_true] = 1

    grad_input = np.subtract(y_true_bool, softmax) / batch_size
    return grad_input
