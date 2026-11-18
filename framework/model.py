"""
U-Net model definition.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers as kl


def UNet(input_shape,
         num_classes=10,
         output_activation='softmax',
         num_layers=4):
    """
    Creates a U-Net model (Ronneberger et al 2015)
    Architecture adapted from github.com/karolzak/keras-unet/master/keras_unet/models/satellite_unet.py
    """

    def bn_conv_relu(input, filters, **conv2d_kwargs):
        x = kl.BatchNormalization()(input)
        x = kl.Conv2D(filters, activation='relu', **conv2d_kwargs)(x)
        return x

    def bn_upconv_relu(input, filters, **conv2d_transpose_kwargs):
        x = kl.BatchNormalization()(input)
        x = kl.Conv2DTranspose(filters, activation='relu', **conv2d_transpose_kwargs)(x)
        return x

    inputs = kl.Input(input_shape)

    # number of filters in a convolution in the contrastive path (constant)
    filters = 64
    # number of filters in a convolution in the dilative path (constant)
    upconv_filters = 96

    conv2d_kwargs = {
        'kernel_size': (3,3),
        'strides': (1,1),
        'padding': 'same',
        'kernel_initializer': 'he_normal'
    }
    conv2d_transpose_kwargs = {
        'kernel_size': (3,3),
        'strides': (2,2),
        'padding': 'same',
        'output_padding': (1,1)
    }
    maxpool2d_kwargs = {
        'pool_size': (2,2),
        'strides': (2,2),
        'padding': 'valid',
    }

    x = kl.Conv2D(filters, activation='relu', **conv2d_kwargs)(inputs)
    c1 = bn_conv_relu(x, filters, **conv2d_kwargs)
    x = bn_conv_relu(c1, filters, **conv2d_kwargs)
    x = kl.MaxPooling2D(**maxpool2d_kwargs)(x)

    down_layers = []

    for _ in range(num_layers):
        x = bn_conv_relu(x, filters, **conv2d_kwargs)
        x = bn_conv_relu(x, filters, **conv2d_kwargs)
        down_layers.append(x)
        x = bn_conv_relu(x, filters, **conv2d_kwargs)
        x = kl.MaxPooling2D(**maxpool2d_kwargs)(x)

    x = bn_conv_relu(x, filters, **conv2d_kwargs)
    x = bn_conv_relu(x, filters, **conv2d_kwargs)
    x = bn_upconv_relu(x, filters, **conv2d_transpose_kwargs)

    for conv in reversed(down_layers):
        x = kl.concatenate([x, conv])
        x = bn_conv_relu(x, upconv_filters, **conv2d_kwargs)
        x = bn_conv_relu(x, filters, **conv2d_kwargs)
        x = bn_upconv_relu(x, filters, **conv2d_transpose_kwargs)

    x = kl.concatenate([x, c1])
    x = bn_conv_relu(x, upconv_filters, **conv2d_kwargs)
    x = bn_conv_relu(x, filters, **conv2d_kwargs)

    outputs = kl.Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), activation=output_activation, padding='valid') (x)

    model = Model(inputs=[inputs], outputs=[outputs], name='unet')
    return model



def _mobilenetv2_4ch(input_shape, alpha=1.0, pretrained=None):
    """MobileNetV2 encoder that accepts 4-channel (R-G-B-NIR) inputs.

    If pretrained == 'imagenet', the ImageNet weights (natural RGB photos, *not* Sentinel-2)
    are copied into the 4-channel network: the kernels of the first convolution for R, G, B
    are kept and the NIR kernel is initialised with the mean of the RGB kernels.
    If pretrained is None, the encoder is trained from scratch.
    """
    MobileNetV2 = tf.keras.applications.MobileNetV2
    encoder = MobileNetV2(input_shape=input_shape, alpha=alpha, include_top=False, weights=None)
    if pretrained == 'imagenet':
        rgb_shape = tuple(input_shape[:2]) + (3,)
        ref = MobileNetV2(input_shape=rgb_shape, alpha=alpha, include_top=False, weights='imagenet')
        for layer, ref_layer in zip(encoder.layers, ref.layers):
            ref_weights = ref_layer.get_weights()
            if not ref_weights:
                continue
            if layer.name == 'Conv1':
                kernel = ref_weights[0]  # (3, 3, 3, filters)
                nir = kernel.mean(axis=2, keepdims=True)
                ref_weights = [np.concatenate([kernel, nir], axis=2)] + ref_weights[1:]
            layer.set_weights(ref_weights)
        del ref
    elif pretrained is not None:
        raise ValueError(f"Unknown pretrained option '{pretrained}' (use 'imagenet' or null)")
    return encoder


def DeepLabV3Plus(input_shape,
                  num_classes=10,
                  output_activation='softmax',
                  backbone_alpha=1.0,
                  pretrained=None,
                  aspp_filters=128,
                  aspp_rates=(3, 6, 9),
                  decoder_filters=128,
                  bn_momentum=0.9,
                  input_mean=None,
                  input_variance=None):
    """
    Lightweight DeepLabV3+ (Chen et al. 2018) with a MobileNetV2 encoder (Sandler et al. 2018).

    - encoder: MobileNetV2 truncated at output stride 16 (16x16 features for a 256x256 input)
    - ASPP with depthwise-separable atrous convolutions (rates adapted to 16x16 feature maps)
    - decoder fusing the stride-4 low-level features, then bilinear upsampling to full resolution
    About 1.1M parameters with the default arguments.

    bn_momentum: momentum of all the BatchNorm layers. Keras' MobileNetV2 uses 0.999, which makes
    the moving statistics (used for validation / inference) converge far too slowly for short
    trainings: the model then predicts a single class on the validation images.
    input_mean, input_variance: per-channel statistics of the (normalized) training images, used
    to standardize the inputs. They are stored in the model, so inference needs nothing else.
    """
    def sep_conv_bn_relu(x, filters, dilation_rate=1, name=None):
        x = kl.SeparableConv2D(filters, 3, padding='same', dilation_rate=dilation_rate,
                               use_bias=False, name=name)(x)
        x = kl.BatchNormalization()(x)
        return kl.ReLU()(x)

    def conv1x1_bn_relu(x, filters):
        x = kl.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = kl.BatchNormalization()(x)
        return kl.ReLU()(x)

    inputs = kl.Input(input_shape)
    # standardize every band with the statistics of the training set. The inputs are divided by
    # the max of the train set, so their values are tiny (variance ~1e-3): a BatchNorm here would
    # need hundreds of steps before its moving statistics are usable for validation.
    if input_mean is None or input_variance is None:
        input_mean, input_variance = [0.] * input_shape[-1], [1.] * input_shape[-1]
    x = kl.Normalization(mean=list(map(float, input_mean)), variance=list(map(float, input_variance)),
                         name='input_standardization')(inputs)

    encoder = _mobilenetv2_4ch(input_shape, alpha=backbone_alpha, pretrained=pretrained)
    features = tf.keras.Model(
        encoder.input,
        [encoder.get_layer('block_3_expand_relu').output,    # stride 4  -> 64x64
         encoder.get_layer('block_13_expand_relu').output],  # stride 16 -> 16x16
        name='mobilenetv2_encoder'
    )
    low_level, high_level = features(x)

    # Atrous Spatial Pyramid Pooling
    h, w = high_level.shape[1], high_level.shape[2]
    pool = kl.GlobalAveragePooling2D(keepdims=True)(high_level)
    pool = conv1x1_bn_relu(pool, aspp_filters)
    pool = kl.UpSampling2D(size=(h, w), interpolation='bilinear')(pool)
    branches = [pool, conv1x1_bn_relu(high_level, aspp_filters)]
    branches += [sep_conv_bn_relu(high_level, aspp_filters, dilation_rate=r) for r in aspp_rates]
    x = kl.Concatenate()(branches)
    x = conv1x1_bn_relu(x, aspp_filters)
    x = kl.Dropout(0.1)(x)

    # Decoder
    x = kl.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)  # 16x16 -> 64x64
    low_level = conv1x1_bn_relu(low_level, 48)
    x = kl.Concatenate()([x, low_level])
    x = sep_conv_bn_relu(x, decoder_filters)
    x = sep_conv_bn_relu(x, decoder_filters)
    x = kl.Conv2D(num_classes, 1, padding='same')(x)
    x = kl.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)  # 64x64 -> 256x256
    outputs = kl.Activation(output_activation, dtype='float32')(x)

    model = Model(inputs=[inputs], outputs=[outputs], name='deeplabv3plus_mobilenetv2')
    for layer in model.submodules:
        if isinstance(layer, kl.BatchNormalization):
            layer.momentum = bn_momentum
    return model


def build_model(name, input_shape, num_classes, **kwargs):
    """Build a model from its name (as given in the YAML config)."""
    models = {
        'unet': UNet,
        'deeplabv3plus': DeepLabV3Plus,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}', choose among {list(models)}")
    return models[name](input_shape=input_shape, num_classes=num_classes, **(kwargs or {}))


if __name__ == '__main__':
    # Test
    unet_kwargs = dict(
        input_shape=(256, 256, 4),
        num_classes=10,
        output_activation='softmax',
        num_layers=2
    )
    print(f"Creating U-Net with arguments: {unet_kwargs}")
    model = UNet(**unet_kwargs)
    print("Summary:")
    print(model.summary())

    input_batch = tf.random.normal((1, 256, 256, 4), name='random_normal_input')
    output = model(input_batch)
    print(output.shape)

    print("Creating DeepLabV3+ (MobileNetV2 encoder, from scratch)")
    model = DeepLabV3Plus(input_shape=(256, 256, 4), num_classes=10)
    print(f"Parameters: {model.count_params():,}")
    print(model(input_batch).shape)
