import numpy as np
import cv2
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------
# Quality Control
# -------------------------
def check_image_quality(image):
    """Checks if the image is blurry using the Variance of Laplacian method."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Threshold for blurriness (typical values between 60-100)
    is_blurry = variance < 70
    return is_blurry, variance

# -------------------------
# Image Preprocessing
# -------------------------
def prepare_image(image, img_size):
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# -------------------------
# SHAP Explainability
# -------------------------
def get_shap_explanation(image_array, model):
    """Generates SHAP values for a single image."""
    # Use a background of zeros for the explainer (GradientExplainer is faster for deep learning)
    background = np.zeros((1, 64, 64, 3))
    explainer = shap.GradientExplainer(model, background)
    
    # Calculate SHAP values for the image
    shap_values = explainer.shap_values(image_array)
    
    # SHAP returns a list for multi-output models, or a single array for binary
    # In binary classification (sigmoid), it usually returns a single array
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    # Generate the SHAP image plot as a Matplotlib figure
    # We need to reshape for the plot
    plt.figure(figsize=(8, 8))
    # Note: shap.image_plot handles normalization and visualization
    shap.image_plot(shap_values, image_array, show=False)
    
    # Capture the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


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
