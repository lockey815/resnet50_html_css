# ==========================================================
# CONVERT best_resnet50.keras  ->  TensorFlow.js (tfjs_model/)
# Run in Google Colab. "# %%" = one cell. Two parts, two sessions.
# ==========================================================

# ---------------- PART A ----------------
# Use the session/runtime where your model loads normally.
# Upload best_resnet50.keras first (left sidebar > Files).

# %% A1: export the weights
import numpy as np, tensorflow as tf

model = tf.keras.models.load_model("best_resnet50.keras", compile=False)
size = model.input_shape[1]
weights = model.get_weights()

x = (np.random.rand(1, size, size, 3) * 255).astype("float32")
x_pre = tf.keras.applications.resnet50.preprocess_input(x.copy())
y = model.predict(x_pre)                      # reference output for checking

np.savez("weights.npz", size=size, x=x_pre, y=y,
         **{f"w{i}": w for i, w in enumerate(weights)})
print("Saved weights.npz | input size:", size, "| tensors:", len(weights))

# Download weights.npz (left sidebar > Files > right-click > Download)

# ---------------- PART B ----------------
# Runtime > Disconnect and delete runtime, then open a FRESH runtime.
# Upload weights.npz. Run these cells IN ORDER, B1 first.

# %% B1: install + legacy Keras (must run BEFORE importing tensorflow)
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# !pip install -q tensorflowjs tf_keras

# %% B2: rebuild the network and load the weights
import re, numpy as np, tensorflow as tf

d = np.load("weights.npz")
size = int(d["size"])
keys = sorted([k for k in d.files if re.fullmatch(r"w\d+", k)],
              key=lambda k: int(k[1:]))
weights = [d[k] for k in keys]

base = tf.keras.applications.ResNet50(
    include_top=False, weights=None, input_shape=(size, size, 3))
inp = tf.keras.Input((size, size, 3))
h = base(inp)
h = tf.keras.layers.GlobalAveragePooling2D()(h)
h = tf.keras.layers.Dense(128, activation="relu")(h)
h = tf.keras.layers.Dropout(0.5)(h)
out = tf.keras.layers.Dense(2, activation="softmax")(h)
m = tf.keras.Model(inp, out)

m.set_weights(weights)

# Check: the rebuilt model must match the original
diff = float(np.abs(m.predict(d["x"]) - d["y"]).max())
print("Max difference vs original:", diff, "(should be near 0)")
m.save("tfjs_src.h5")

# %% B3: convert to TensorFlow.js (float16 = smaller download)
# !tensorflowjs_converter --input_format=keras --quantize_float16 tfjs_src.h5 tfjs_model
# !zip -r tfjs_model.zip tfjs_model
# Download tfjs_model.zip, unzip it, and put its files in the site's tfjs_model/ folder.
