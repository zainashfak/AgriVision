import numpy as np
import cv2
import tensorflow as tf
import keras


def gradcam(model, img_rgb, class_idx=None, size=224):
    """Return the image with a Grad-CAM heatmap overlay (RGB uint8)."""
    x = np.array(cv2.resize(img_rgb, (size, size)), dtype="float32")[None]
    x = tf.convert_to_tensor(keras.applications.mobilenet_v2.preprocess_input(x))

    # the MobileNetV2 base is the nested model inside our classifier
    base = next(l for l in model.layers
                if isinstance(l, keras.Model) and "mobilenet" in l.name.lower())
    head = model.layers[model.layers.index(base) + 1:]

    with tf.GradientTape() as tape:
        tape.watch(x)
        conv = base(x)
        y = conv
        for layer in head:
            y = layer(y)
        if class_idx is None:
            class_idx = int(tf.argmax(y[0]))
        score = y[:, class_idx]

    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))

    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8(0.55 * img_rgb + 0.45 * heat)