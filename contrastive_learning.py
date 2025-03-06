import tensorflow as tf

class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def call(self, anchor, positive, negatives):
        pos_similarity = tf.exp(tf.reduce_sum(anchor * positive, axis=-1) / self.temperature)
        neg_similarity = tf.exp(tf.reduce_sum(anchor * negatives, axis=-1) / self.temperature)
        
        loss = -tf.math.log(pos_similarity / (pos_similarity + tf.reduce_sum(neg_similarity, axis=-1)))
        return tf.reduce_mean(loss)
