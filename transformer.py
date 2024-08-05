
# transformer.py

# packages
import os
import time
import argparse
import logging
from datetime import datetime
from collections import deque
import pandas as pd
import numpy as np
import psycopg2
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# from files
from load_data import load_and_merge_data, label_data_split



def setup_logger():
    # Create a logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Generate a filename based on the current datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_log_{current_time}.log"
    
    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        # Define the layers used in the Transformer block
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.multi_head_attention = MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )
        self.dropout1 = Dropout(self.dropout)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.dense_ff = Dense(self.ff_dim, activation='relu')
        self.dropout2 = Dropout(self.dropout)
        self.dense_output = Dense(ff_dim)

    def call(self, inputs, training=False):
        # Apply layer normalization and multi-head attention
        x = self.layer_norm1(inputs)
        x = self.multi_head_attention(x, x)
        x = self.dropout1(x, training=training)
        res = x + inputs

        # Apply feed-forward network and residual connection
        x = self.layer_norm2(res)
        x = self.dense_ff(x)
        x = self.dropout2(x, training=training)
        x = self.dense_output(x)

        # Return final output with another residual connection
        return x + res

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Define hyperparameters
        head_size = 492
        num_heads = 4
        ff_dim = 492
        num_transformer_blocks = 4
        mlp_units = [128, 128]
        dropout = 0.2
        mlp_dropout = 0.2

        # Build the transformer blocks
        self.transformer_blocks = [
            TransformerBlock(head_size, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_blocks)
        ]

        # Add GlobalAveragePooling1D and MLP layers
        self.pooling = GlobalAveragePooling1D()
        self.mlp_layers = [Dense(units, activation='relu') for units in mlp_units]
        self.mlp_dropout_layers = [Dropout(mlp_dropout) for _ in mlp_units]

        # Output layer with 48 units for multi-step prediction
        self.output_layer = Dense(48, activation="linear")  # 4 intervals

    def call(self, inputs, training=False):
        # Forward pass through transformer blocks
        x = inputs
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)

        # Pooling and MLP layers
        x = self.pooling(x)
        for layer, dropout in zip(self.mlp_layers, self.mlp_dropout_layers):
            x = layer(x)
            if training:
                x = dropout(x)

        # Output
        return self.output_layer(x)

def create_dataset(product_ids, use_recent):
    dataset = tf.data.Dataset.from_generator(
        lambda: load_data(product_ids, use_recent),
        output_signature=(
            tf.TensorSpec(shape=(1000, 492), dtype=tf.float32),
            tf.TensorSpec(shape=(1, 48), dtype=tf.float32)
        )
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def load_data(product_ids, use_recent = False):
    #logger.info(f"load_data: Loading and preparing data for {product_ids}")
    
    # First, align and merge data from all tables
    data = load_and_merge_data(product_ids, use_recent)
    #logger.info("load data: dimensions of combined data are", data.shape)
    
    # Load the aligned data from the saved CSV file
    horizons = [10, 20, 100, 200]
    data_generator = label_data_split(data, product_ids, horizons)
    
    for X_train, y in data_generator():

        scaler = tf.keras.layers.experimental.preprocessing.Normalization()
        scaler.adapt(X_train)
        
        yield X_train, y


def make_predictions(model, X_tensor, scaler):
    logger.info("make_predictions: Making predictions")
    
    # Scale the input tensor
    X_scaled = scaler(X_tensor)

    # Make predictions using the model
    predictions = model.predict(X_scaled)

    # Since the model predicts all intervals at once, we don't need to loop over intervals
    logger.info("make_predictions: Predictions made successfully")
    
    return predictions


# Callback to simulate ReduceLROnPlateau behavior
class CustomReduceLROnPlateau(Callback):
    def __init__(self, monitor='loss', factor=2, patience=5, min_lr=1e-6, verbose=1):
        super(CustomReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best = np.Inf
        self.wait = 0
        self.model = None

    def set_model(self, model):
        """Manually set the model for the callback."""
        self.model = model
        if self.model is None:
            raise ValueError("Model cannot be None. Ensure it is set before training starts.")

    def on_batch_end(self, logs=None):
        if self.model is None:
            logger.error("Model not set in callback. Skipping learning rate update.")
            return
        
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            logger.warning(f"Warning: Metric '{self.monitor}' not found in logs. Skipping learning rate update.")
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                    if self.verbose > 0:
                        logger.info(f"CustomReduceLROnPlateau: Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                self.wait = 0

def main_online(use_recent=False, plot_graph=False):
    product_ids = [
        "BTC-USD", "ETH-USD", "MATIC-USD", "SOL-USD", "DAI-USD", 
        "DOGE-USD", "ETH-BTC", "MATIC-BTC", "SOL-BTC", "DOGE-BTC", 
        "SOL-ETH", "ETH-DAI"
    ]
    logger.info(f"main_online: Starting online process for {product_ids}")

    # Initialize the model
    model = MyModel()
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Load weights if available
    checkpoint_path = 'checkpoints/model_weights.h5'
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading existing weights from {checkpoint_path}")
        try:
            model.load_weights(checkpoint_path)
        except Exception as e:
            logger.info(f"Error loading weights: {e}")

    # Create dataset
    dataset = create_dataset(product_ids, use_recent)

    # Initialize variables for custom checkpointing
    best_loss = np.inf

    # Instantiate custom ReduceLROnPlateau callback
    reduce_lr_callback = CustomReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    reduce_lr_callback.set_model(model)  # Manually set the model

    # Dynamic batch size logic
    batch_data = []
    batch_labels = []
    start_time = time.time()

    # Initialize variables for tracking loss per batch
    batch_losses = []
    batch_maes = []
    if plot_graph:
        #logger.info("plot graph if condition")
        # Enable interactive mode
        plt.ion()

        # Create a figure and primary axis for Loss
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_title('Loss and MAE vs. Batch Number')
        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('Loss (Log Scale)', color='blue')
        ax1.set_yscale('log')  # Set the loss y-axis to logarithmic scale
        loss_line, = ax1.plot([], [], label='Batch Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        # Secondary axis for MAE
        ax2 = ax1.twinx()
        ax2.set_ylabel('MAE', color='orange')
        mae_line, = ax2.plot([], [], label='Batch MAE', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Add legends
        lines = [loss_line, mae_line]
        ax1.legend(lines, [line.get_label() for line in lines])

        # Slider for adjusting the x-axis scale
        ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Scale', valmin=1, valmax=100, valinit=10, valstep=1)

        # Function to update the x-axis scale based on slider
        def update(val):
            scale = slider.val
            ax1.set_xlim(left=max(0, len(batch_losses) - scale), right=len(batch_losses))
            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Show the plot without blocking
        plt.show(block=False)

    batches_trained = 0
    # Training loop
    for X_sample, y_label in dataset:
        try:
            # Append new sample to batch
            batch_data.append(X_sample)
            batch_labels.append(y_label)

            # Check elapsed time
            elapsed_time = time.time() - start_time

            # If one second has passed, train on the collected batch
            if elapsed_time >= 1.0 or len(batch_labels) >= 15:
                # Convert lists to numpy arrays
                batch_data_np = tf.stack(batch_data)
                batch_labels_np = tf.stack(batch_labels)

                batch_start_time = time.time()
                # Train on the current batch
                try:
                    loss_dict = model.train_on_batch(batch_data_np, batch_labels_np, return_dict=True)
                except Exception as e:
                    logger.info(f"Error during model.train_on_batch: {e}")
                    logger.info(f"Model state: {model}")
                    continue
                #logger.info(f"Model after training: {model}")

                current_loss = loss_dict["loss"]

                batch_train_time = time.time() - batch_start_time
                batches_trained += 1
                batch_losses.append(current_loss)
                batch_maes.append(loss_dict["mae"])

                logger.info(f"Batch {batches_trained} trained in {batch_train_time}")
                if plot_graph:
                    # Update the plot with new loss and MAE data
                    loss_line.set_xdata(range(len(batch_losses)))
                    loss_line.set_ydata(batch_losses)
                    mae_line.set_xdata(range(len(batch_maes)))
                    mae_line.set_ydata(batch_maes)
                    ax1.relim()  # Recalculate limits for loss
                    ax2.relim()  # Recalculate limits for MAE
                    ax1.autoscale_view()  # Autoscale for loss
                    ax2.autoscale_view()  # Autoscale for MAE
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.01)  # Pause to update the plot

                # logger.info loss and metrics for the current batch
                loss_output = ', '.join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                logger.info(f"main_online: loss - {loss_output} - Batch Size: {len(batch_data)}")

                # Custom checkpointing logic: save best model weights
                if current_loss < best_loss:
                    logger.info(f"main_online: New best loss {current_loss:.4f} found, saving model weights.")
                    best_loss = current_loss
                    #model.save_weights(checkpoint_path)

                # Custom ReduceLROnPlateau logic: adjust learning rate
                #reduce_lr_callback.on_batch_end(logs=loss_dict)
                if batches_trained % 50 == 0:
                    logger.info(f"main_online: {batches_trained} batches trained, saving weights")
                    model.save_weights(checkpoint_path)
                # Reset batch data and start time
                batch_data = []
                batch_labels = []
                start_time = time.time()
                
        except Exception as e:
            logger.info(f"Error during training: {e}")

    logger.info(f"main_online: Process completed successfully for all {product_ids}")

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Script started")
    
    parser = argparse.ArgumentParser(description="Run the transformer model with options.")
    parser.add_argument('--recent', action='store_true', help='Use the most recent data file')
    parser.add_argument('--graph', action='store_true', help='Plot loss graph after training')
    args = parser.parse_args()

    logger.info(f"Arguments: recent={args.recent}, graph={args.graph}")
    main_online(use_recent=args.recent, plot_graph=args.graph)

