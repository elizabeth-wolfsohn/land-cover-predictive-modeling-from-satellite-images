import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def load_imagenet_weights_n_channels(model, n_channels, input_hw):
    """Copie les poids ImageNet dans un ResNet50 à n canaux."""
    ref = ResNet50(include_top=False, weights="imagenet",
                   input_shape=(*input_hw, 3))

    for layer in model.layers:
        if layer.name == "conv1_conv":
            w, b = ref.get_layer("conv1_conv").get_weights()   # w : (7, 7, 3, 64)
            extra = n_channels - 3
            mean_w = w.mean(axis=2, keepdims=True)             # moyenne des canaux RGB
            w_new = np.concatenate([w, np.repeat(mean_w, extra, axis=2)], axis=2)
            w_new *= 3.0 / n_channels                          # garde la même amplitude d'activation
            layer.set_weights([w_new, b])
        elif layer.weights:
            layer.set_weights(ref.get_layer(layer.name).get_weights())


def build_resnet50_unet(input_shape=(256, 256, 4),
                        num_classes=10,
                        output_activation="softmax"):
    inputs = Input(input_shape)

    # weights=None : on peut avoir 4 canaux, les poids sont chargés ensuite
    resnet50 = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    load_imagenet_weights_n_channels(resnet50, input_shape[-1], input_shape[:2])

    # Encodeur
    s1 = inputs
    s2 = resnet50.get_layer("conv1_relu").output
    s3 = resnet50.get_layer("conv2_block3_out").output
    s4 = resnet50.get_layer("conv3_block4_out").output
    b1 = resnet50.get_layer("conv4_block6_out").output

    # Décodeur
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation=output_activation)(d4)
    return Model(inputs, outputs, name="ResNet50_U-Net")


if __name__ == "__main__":
    model = build_resnet50_unet((256, 256, 4), num_classes=10, output_activation="softmax")
    model.summary()