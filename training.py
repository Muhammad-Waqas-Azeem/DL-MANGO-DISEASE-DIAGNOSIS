import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# Load datasets (already batched)
train_dataset = tf.data.Dataset.load("train_dataset")
val_dataset = tf.data.Dataset.load("val_dataset")

# Ensure dataset labels are correct type
def fix_labels(images, labels):
    return images, tf.cast(labels, tf.int32)

# Apply label fix, cache, prefetch (NO extra batching)
train_dataset = train_dataset.map(fix_labels).cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(fix_labels).cache().prefetch(tf.data.AUTOTUNE)

# Debugging shape
for sample_images, sample_labels in train_dataset.take(1):
    print("Train batch shape:", sample_images.shape)  # Should be (BATCH_SIZE, 256, 256, 3)

# Define improved CNN model
inputs = tf.keras.Input(shape=(256, 256, 3))

x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(8, activation='softmax')(x)  # 8 classes

model = models.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# Train model
EPOCHS = 20
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, early_stopping, reduce_lr]
)

# Save final model
model.save("final_model.keras")
print("✅ Final model saved as 'final_model.keras'")

