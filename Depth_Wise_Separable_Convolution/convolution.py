import numpy as np

#C : Number of input channels
# H: Height of input
#W:  Width of input
#R_H : Kernel Height
#R_W : Kernel Width
#F : Number of features
# def apply_kernel(img, kernel):
#     return np.sum(np.multiply(img, kernel))
def conv2d_multi_channel(input, w):
    """Two-dimensional convolution with multiple channels.

        Uses SAME padding with 0s, a stride of 1 and no dilation.

        input: input array with shape (height, width, in_depth)
        w: filter array with shape (fd, fd, in_depth, out_depth) with odd fd.
           in_depth is the number of input channels, and has the be the same as
           input's in_depth; out_depth is the number of output channels.

        Returns a result with shape (height, width, out_depth).
        """


    # padw = w.shape[0] // 2
    # padded_input = np.pad(input,
    #                       pad_width=((padw, padw), (padw, padw), (0, 0)),
    #                       mode='constant',
    #                       constant_values=0)
    in_depth,height, width = input.shape
    fc,fh,fw = w.shape
    output = np.zeros((fc,height-fh+1, width-fw+1))

    for out_c in range(fc):
        # For each output channel, perform 2d convolution summed across all
        # input channels.
        for i in range(height-fh+1):
            for j in range(width-fw+1):
                # Now the inner loop also works across all input channels.
                for c in range(in_depth):
                    # for fi in range(fh):
                    #     for fj in range(fw):
                            input_mat = input[c][i:i+fh,j:j+fw]
                            # print(np.sum(input_mat*w[c]))
                            output[out_c,i,j] = output[out_c,i,j] + np.sum(input_mat*w[c])
                            # print(output)
                            # output[i, j, out_c] += (
                            #         padded_input[i + fi, j + fj, c] * w_element)
    return output



def depthwise_conv2d(input, w):
    """Two-dimensional depthwise convolution.

    Uses SAME padding with 0s, a stride of 1 and no dilation. A single output
    channel is used per input channel (channel_multiplier=1).

    input: input array with shape (height, width, in_depth)
    w: filter array with shape (fd, fd, in_depth)

    Returns a result with shape (height, width, in_depth).
    """
    height, width, in_depth = input.shape
    fc, fh, fw = w.shape
    output = np.zeros((height-fh+1, width-fw+1, in_depth))

    for c in range(in_depth):
        # For each input channel separately, apply its corresponsing filter
        # to the input.
        for i in range(height-fh+1):
            for j in range(width-fw+1):
                input_mat = input[c][i:i + fh, j:j + fw]
                # print(np.sum(input_mat*w[c]))
                output[i, j,c] = output[i, j,c] + np.sum(input_mat * w[c])
    return output

#C : Number of input channels
# H: Height of input
#W:  Width of input
#R_H : Kernel Height
#R_W : Kernel Width
#F : Number of features

def separable_conv2d(input,C, H, W, R_H, R_W, F, input_t, dweights, pweights):
    """Depthwise separable convolution.

        Performs 2d depthwise convolution with w_depth, and then applies a pointwise
        1x1 convolution with w_pointwise on the result.

        Uses SAME padding with 0s, a stride of 1 and no dilation. A single output
        channel is used per input channel (channel_multiplier=1) in w_depth.

        input: input array with shape (height, width, in_depth)
        w_depth: depthwise filter array with shape (fd, fd, in_depth)
        w_pointwise: pointwise filter array with shape (in_depth, out_depth)

        Returns a result with shape (height, width, out_depth).
        """
    # First run the depthwise convolution. Its result has the same shape as
    # input.
    depthwise_result = depthwise_conv2d(input,filter)
    # print(depthwise_result.shape)
    height, width, in_depth = depthwise_result.shape
    print(pweights.shape)
    p_ch,p_h,p_w = pweights.shape
    out_ch = 3

    final_output = np.zeros((p_ch,height, width))
    # print("------------------------------------")
    # print(depthwise_result[0])
    # print(pweights)
    # print("-------------------------------------")

    for p_c in range(p_ch):
        output = np.zeros((height, width))
        for i in range(in_depth):
            input_mat = depthwise_result[:,:,i]
            output =  output + input_mat * pweights[p_c]
        final_output[p_c] = output

    return final_output


red   = np.array([1]*9).reshape((3,3))
green = np.array([100]*9).reshape((3,3))
blue  = np.array([10000]*9).reshape((3,3))

img = np.stack([red, green, blue])
# print(img.shape)
# img = np.expand_dims(img, axis=0)
# print(img.shape)
# print(img[0,:,:])
# print(img[0][0][0])
filter = np.ones((3,2,2))
p_filter = np.ones((3,1,1))
o = conv2d_multi_channel(img,filter)
print(o)
print(o.shape)
print("---------------------------")
output_matrix = separable_conv2d(img,filter,3,3,3,2,2,1,1,p_filter)
print(output_matrix)
print(output_matrix.shape)

