import numpy as np
import cv2
import tensorflow as tf

# -------------------------
# Image Preprocessing
# -------------------------
def prepare_image(image, img_size):
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# -------------------------
# Grad-CAM
# -------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        # Automatically find last conv layer
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break
    
    # If no conv layer found, return zeros
    if last_conv_layer_name is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    # Ensure model has been called with the correct shape
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    try:
        # Create a new functional model that mimics the original model
        # but outputs both the last conv layer activations and the final prediction.
        inputs = tf.keras.Input(shape=model.input_shape[1:])
        x = inputs
        conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
        
        if conv_output is None:
             return np.zeros((img_array.shape[1], img_array.shape[2]))
             
        grad_model = tf.keras.models.Model(inputs, [conv_output, x])
    except Exception:
        # Fallback
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        tape.watch(conv_outputs)
        
        # Binary classification (sigmoid) - use the probability of being Parasitized
        loss = predictions[:, 0]

    # Extract gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Check if grads is None
    if grads is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    # Mean intensity of the gradient over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature map by the gradient
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to emphasize important regions and Normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val
        
    return heatmap.numpy()


# -------------------------
# Overlay Heatmap
# -------------------------
def overlay_heatmap(original_img, heatmap, alpha=0.4):
    # If heatmap is empty/zeros, return original
    if np.max(heatmap) == 0:
        return original_img
        
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img