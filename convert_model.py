import tensorflow as tf

print("Loading .keras model...")
model = tf.keras.models.load_model("saved_model/mango_model.keras")

print("Saving as .h5 model...")
model.save("saved_model/mango_model.h5")

print("Conversion complete!")
