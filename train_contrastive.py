import tensorflow as tf
from model.encoder import build_encoder
from model.contrastive_learning import ContrastiveLoss
from data_processing.data_loader import load_seismic_data

# Load data
seismic_data = load_seismic_data("data/")

# Define encoder
encoder = build_encoder(input_shape=(250, 1))

# Contrastive loss function
contrastive_loss = ContrastiveLoss()

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training Loop
for epoch in range(50):
    with tf.GradientTape() as tape:
        # Generate augmented pairs
        anchor, positive, negatives = generate_contrastive_pairs(seismic_data)

        # Encode signals
        anchor_encoded = encoder(anchor)
        positive_encoded = encoder(positive)
        negatives_encoded = encoder(negatives)

        # Compute contrastive loss
        loss = contrastive_loss(anchor_encoded, positive_encoded, negatives_encoded)

    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))

    print(f"Epoch {epoch + 1}: Loss = {loss.numpy():.4f}")
