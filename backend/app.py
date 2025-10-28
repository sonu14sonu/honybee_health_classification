from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import Model, Input
from math import ceil

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'bee_health_model.keras')

# --- Load trained model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Extract base MobileNet from the trained model ---
base_model = model.layers[0]  # assuming base_model is the first layer in Sequential

# --- Pick only Conv2D layers from base_model for feature maps ---
conv_layers = [layer.output for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
layer_names = [layer.name for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
intermediate_layer_model = Model(inputs=base_model.input, outputs=conv_layers)

# === Bee Health Class Names ===
class_names = [
    'Varroa, Small Hive Beetles',
    'ant problems',
    'few varrao, hive beetles',
    'healthy',
    'hive being robbed',
    'missing queen'
]

# === Flask App Setup ===
app = Flask(__name__, static_folder='../frontend', static_url_path='')

# === Utility Functions ===
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def plot_feature_map_grid(fmap, layer_name, max_cols=8):
    """Plots all channels of a feature map in a single grid image."""
    n_channels = fmap.shape[-1]
    cols = min(max_cols, n_channels)
    rows = ceil(n_channels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = np.array(axes).reshape(-1)

    for i in range(n_channels):
        fmap_channel = fmap[0, :, :, i]
        fmap_channel -= fmap_channel.min()
        if fmap_channel.max() != 0:
            fmap_channel /= fmap_channel.max()
        axes[i].imshow(fmap_channel, cmap='viridis')
        axes[i].axis('off')

    # Hide unused axes
    for j in range(n_channels, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Layer {layer_name} Feature Maps', fontsize=16)
    fig.tight_layout()
    return fig_to_base64(fig)

# === Routes ===
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        original_img = Image.open(io.BytesIO(file.read())).convert('RGB')
        preprocessing_steps = []

        # Original Image
        preprocessing_steps.append({'name': 'Original Image', 'image': image_to_base64(original_img)})

        # Resize Image
        resized_img = original_img.resize((128, 128))
        preprocessing_steps.append({'name': 'Resized Image (128x128)', 'image': image_to_base64(resized_img)})

        # Normalize
        img_array = tf.keras.preprocessing.image.img_to_array(resized_img) / 255.0
        batched_array = np.expand_dims(img_array, axis=0)

        # --- Intermediate Feature Maps ---
        intermediate_outputs = intermediate_layer_model.predict(batched_array)
        if not isinstance(intermediate_outputs, list):
            intermediate_outputs = [intermediate_outputs]

        for i, fmap in enumerate(intermediate_outputs[:10]):  # limit to first 10 conv layers
            if fmap.ndim == 4:
                img_base64 = plot_feature_map_grid(fmap, layer_names[i])
                preprocessing_steps.append({'name': f'Conv Layer {layer_names[i]}', 'image': img_base64})

        # --- Final Prediction ---
        predictions = model.predict(batched_array)
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])

        # --- Prediction Probability Plot ---
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(class_names, predictions[0])
        bars[predicted_class_index].set_color('red')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.tick_params(axis='x', labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_ha('right')
        fig.tight_layout()
        preprocessing_steps.append({'name': 'Prediction Probabilities', 'image': fig_to_base64(fig)})

        return jsonify({'class': class_names[predicted_class_index],
                        'confidence': confidence,
                        'preprocessing_steps': preprocessing_steps})

    except Exception as e:
        import traceback
        print("❌ Error during classification:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/debug', methods=['GET'])
def debug_model():
    dummy = np.random.rand(1, 128, 128, 3).astype(np.float32)
    pred = model.predict(dummy)
    return jsonify({"dummy_input_shape": [1, 128, 128, 3],
                    "dummy_output": pred.tolist(),
                    "conv_layers": layer_names})

# === Run Server ===
if __name__ == '__main__':
    os.makedirs('../frontend/images', exist_ok=True)
    port = 5000
    print(f"🚀 Starting Flask server at http://127.0.0.1:{port}")
    app.run(debug=True, host='127.0.0.1', port=port)
