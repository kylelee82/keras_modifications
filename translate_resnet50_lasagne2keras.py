#!/usr/bin/python
import lasagne
from lasagne import layers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD

from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json

from CNN_JPG_Classifier import CNN_JPG_Classifier
from numpy import *
from sklearn import cross_validation
import sys
sys.setrecursionlimit(50000)



##### resnet related layers #####
def build_simple_block(incoming_layer, names,
        	           num_filters, filter_size, stride, pad,
                       	   use_bias=False, nonlin="relu"):
    	"""Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)
    	Parameters:
    	----------
    	incoming_layer : instance of Lasagne layer
        	Parent layer
    	names : list of string
        	Names of the layers in block
    	num_filters : int
        	Number of filters in convolution layer
    	filter_size : int
        	Size of filters in convolution layer
    	stride : int
        	Stride of convolution layer
    	pad : int
        	Padding of convolution layer
    	use_bias : bool
        	Whether to use bias in conlovution layer
    	nonlin : function
        	Nonlinearity type of Nonlinearity layer
    	Returns
    	-------
    	tuple: (net, last_layer_name)
        	net : dict
            	Dictionary with stacked layers
        	last_layer_name : string
            	Last layer name
    	"""
	if pad == 0:
		border_mode = "valid"
	else:
		border_mode = "same"

    	net = []
	net.append((
		names[0],
		Convolution2D(nb_filter=num_filters, nb_row=filter_size, nb_col=filter_size, subsample=(stride,stride), init="he_normal", border_mode=border_mode, name=names[0])(incoming_layer)
		))

	net.append((
		names[1],
		BatchNormalization(mode=0, axis=1,name=names[1])(net[-1][1])
		))

    	if nonlin is not None:
		net.append((
			names[2],
			Activation(nonlin,name=names[2])(net[-1][1])
		))

    	return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                             upscale_factor=4, ix='', left_shapes=[], right_shapes=[]):
    	"""Creates two-branch residual block
    	Parameters:
    	----------
    	incoming_layer : instance of Lasagne layer
        	Parent layer
    	ratio_n_filter : float
        	Scale factor of filter bank at the input of residual block
    	ratio_size : float
        	Scale factor of filter size
    	has_left_branch : bool
        	if True, then left branch contains simple block
    	upscale_factor : float
        	Scale factor of filter bank at the output of residual block
    	ix : int
        	Id of residual block
    	Returns
    	-------
    	tuple: (net, last_layer_name)
        	net : dict
            	Dictionary with stacked layers
        	last_layer_name : string
            	Last layer name
    	"""
    	simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    	net = {}

    	# right branch
    	net_tmp, last_layer_name = build_simple_block(
        	incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        	int(right_shapes[0]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    	net.update(net_tmp)

    	net_tmp, last_layer_name = build_simple_block(
        	net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        	right_shapes[1], 3, 1, 1)
    	net.update(net_tmp)

    	net_tmp, last_layer_name = build_simple_block(
        	net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        	right_shapes[2]*upscale_factor, 1, 1, 0,
        	nonlin=None)
    	net.update(net_tmp)

    	right_tail = net[last_layer_name]
    	left_tail = incoming_layer

    	# left branch
    	if has_left_branch:
        	net_tmp, last_layer_name = build_simple_block(
            		incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            		int(left_shapes[0]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            		nonlin=None)
        	net.update(net_tmp)
        	left_tail = net[last_layer_name]

    	#net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1,name='res%s' % ix)
    	#net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify,name='res%s_relu' % ix)

    	net['res%s' % ix] = merge([left_tail, right_tail], mode="sum",name='res%s' % ix)
    	net['res%s_relu' % ix] = Activation("relu",name='res%s_relu' % ix)(net['res%s' % ix])

    	return net, 'res%s_relu' % ix

def build_resnet50_model(img_rows=224,img_cols=224,color_type=3):
    	net = {}
    	#net['input'] = InputLayer((None, 3, 224, 224),name="input")
	net['input'] = Input(shape=(color_type, img_rows, img_cols),name="input")

    	sub_net, parent_layer_name = build_simple_block(
        	net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        	64, 7, 3, 2, use_bias=True)
    	net.update(sub_net)

    	#net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False,name="pool1")
	net['pool1'] = MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode="valid", name="pool1")(net[parent_layer_name])   
 	block_size = list('abc')
    	parent_layer_name = 'pool1'
        left_shapes = dict()
        right_shapes = dict()
        left_shapes['a']  = [64]
        right_shapes['a'] = [64,64,64]
        right_shapes['b'] = [256,64,64]
        right_shapes['c'] = [256,64,64]
    	for c in block_size:
        	if c == 'a':
            		sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c,left_shapes=left_shapes[c], right_shapes=right_shapes[c])
        	else:
            		sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c, right_shapes=right_shapes[c])
        	net.update(sub_net)

    	block_size = list('abcd')
        left_shapes = dict()
        right_shapes = dict()
        left_shapes['a']  = [256]
        right_shapes['a'] = [256,128,128]
        right_shapes['b'] = [512,128,128]
        right_shapes['c'] = [512,128,128]
        right_shapes['d'] = [512,128,128]
    	for c in block_size:
        	if c == 'a':
            		sub_net, parent_layer_name = build_residual_block(
                		net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c,left_shapes=left_shapes[c], right_shapes=right_shapes[c])
        	else:
            		sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c, right_shapes=right_shapes[c])
        	net.update(sub_net)

    	block_size = list('abcdef')
        left_shapes['a']  = [512]
        right_shapes['a'] = [512,256,256]
        right_shapes['b'] = [1024,256,256]
        right_shapes['c'] = [1024,256,256]
        right_shapes['d'] = [1024,256,256]
        right_shapes['e'] = [1024,256,256]
        right_shapes['f'] = [1024,256,256]

    	for c in block_size:
        	if c == 'a':
            		sub_net, parent_layer_name = build_residual_block(
                		net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c,left_shapes=left_shapes[c], right_shapes=right_shapes[c])
        	else:
            		sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c, right_shapes=right_shapes[c])
        	net.update(sub_net)

    	block_size = list('abc')
        left_shapes['a']  = [1024]
        right_shapes['a'] = [1024,512,512]
        right_shapes['b'] = [2048,512,512]
        right_shapes['c'] = [2048,512,512]

    	for c in block_size:
        	if c == 'a':
            		sub_net, parent_layer_name = build_residual_block(
                		net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c,left_shapes=left_shapes[c], right_shapes=right_shapes[c])
        	else:
            		sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c, right_shapes=right_shapes[c])
        	net.update(sub_net)

    	net['pool5']    = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode="valid",name='pool5')(net[parent_layer_name])
	net['flatten1'] = Flatten()(net['pool5'])
	net['fc1000']   = Dense(output_dim=1000, init="he_normal",name='fc1000')(net['flatten1'])
	net['prob']     = Activation("softmax",name="prob")(net['fc1000'])

	model = Model(input=net['input'], output=net['prob'])
        sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
    	return model

resnet50_keras = build_resnet50_model()

resnet50_lasagne = CNN_JPG_Classifier()
resnet50_lasagne.set_img_zoom(0,450,50,640)
resnet50_lasagne.set_verbosity(True)
resnet50_lasagne.set_grayscale(False)
resnet50_lasagne.augment_training(False)
resnet50_lasagne.augment_training2(False)
resnet50_lasagne.set_resize(True,224,224)
resnet50_lasagne.set_num_classes(1000)
resnet50_lasagne.set_max_train(10000)
resnet50_lasagne.set_pretrain_name("resnet")
resnet50_lasagne.set_zoom_mode(False)
resnet50_lasagne.set_pretrain_model(True)

resnet50_lasagne.define_cnn_resnet50()
resnet50_lasagne.load_external_model("/home/kyle/scripts/kaggle/statefarm/external/resnet50.pkl",with_params=False)

# transfer weights from lasagne to keras
for idx, layer in enumerate(resnet50_lasagne.classifier.layers_.values()):
        params = layer.get_params()
        layer_name = resnet50_lasagne.classifier.layers_[idx]
        if len(layer.params) > 0:
		W = []
		b = []
		gamma = []
		beta = []
		mean = []
		std = []
		layer_keras_name = ""
                for param in layer.params:
                        values = param.get_value()
                        print("Info: Lasagne layer parameter: {}, entries: {} ".format(param,len(values)))
			for layer_keras in resnet50_keras.layers:
				if layer_keras.name == str(param).split(".")[0]:
					type = str(param).split(".")[1]
					if type == "W":
						W = values
					elif type == "b":
						b = values
					elif type == "gamma":
						gamma = values
					elif type == "beta":
						beta = values
					elif type == "mean":
						mean = values
					elif type == "inv_std":
						std = 1./values
					layer_keras_name = layer_keras.name

		if len(W) > 0 and len(b) > 0:
			for layer_keras in resnet50_keras.layers:
				if layer_keras.name == layer_keras_name:
					print("Info: Setting weights and biases for layer {}".format(layer_keras_name))
					layer_keras.set_weights([W,b])			
		elif len(W) > 0:
                        for layer_keras in resnet50_keras.layers:
                                if layer_keras.name == layer_keras_name:
                                        print("Info: Setting weights only for layer {} (bias=0)".format(layer_keras_name))
                                        layer_keras.set_weights([W,zeros(shape=(len(W)))])
		elif len(mean) > 0:
                        for layer_keras in resnet50_keras.layers:
                                if layer_keras.name == layer_keras_name:
                                        print("Info: Setting gamma/beta/mean/std for layer {}".format(layer_keras_name))
					layer_keras.set_weights([gamma,beta,mean,std])
		elif layer_keras_name != "":
                        for layer_keras in resnet50_keras.layers:
                                if layer_keras.name == layer_keras_name:
					print("WARNING: Did not set weights for layer {}.  Expect weights of type {}".format(layer_keras_name, shape(layer_keras.get_weights())))
		else:
			print("WARNING: Layer {} was not found in Keras model".format(layer))

resnet50_keras.save_weights('resnet50.keras.h5')
print("Info: Resnet50 model save to resnet50.keras.h5!")
