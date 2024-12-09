# cli/main.py

import click
import os
import numpy as np
import logging
from models.gan import CryptoGAN
from utils.data_processing import load_data, save_generated_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.command()
@click.option('--input', '-i', default='sample_data/bitcoin_prices.csv',
              help='Path to the input CSV file containing crypto prices.')
@click.option('--output', '-o', default='generated_data/synthetic_prices.csv',
              help='Path to save the generated synthetic data.')
@click.option('--iterations', '-n', default=20000, type=int,
              help='Number of training iterations.')
@click.option('--batch_size', '-b', default=128, type=int,
              help='Batch size for training.')
@click.option('--z_dim', default=24, type=int,
              help='Dimension of the random noise vector.')
@click.option('--hidden_dim', default=24, type=int,
              help='Hidden dimension size for RNN cells.')
@click.option('--num_layers', default=3, type=int,
              help='Number of layers in RNN and transformer.')
@click.option('--d_model', default=24, type=int,
              help='Model dimension for the transformer.')
@click.option('--num_heads', default=2, type=int,
              help='Number of attention heads in the transformer.')
@click.option('--dff', default=128, type=int,
              help='Feed-forward network dimension in the transformer.')
def main(input, output, iterations, batch_size, z_dim, hidden_dim, num_layers, d_model, num_heads, dff):
    """
    High-Frequency Crypto Time-Series Generation CLI Tool using GAN.
    
    Generates synthetic cryptocurrency price data based on real historical data.
    """
    logger.info("Loading original data from %s...", input)
    ori_data = load_data(input)
    
    # Initialize GAN parameters
    parameters = {
        'module': 'gru',
        'hidden_dim': hidden_dim,
        'num_layer': num_layers,
        'iterations': iterations,
        'batch_size': batch_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'dff': dff,
        'z_dim': z_dim,
        'dim': ori_data.shape[-1],  # Number of features
    }
    
    logger.info("Initializing CryptoGAN model with parameters: %s", parameters)
    gan = CryptoGAN(parameters)
    
    logger.info("Starting training for %d iterations...", iterations)
    gan.train(ori_data)
    
    logger.info("Generating synthetic data...")
    generated_data = gan.generate(ori_data)
    
    logger.info("Saving generated data to %s...", output)
    save_generated_data(generated_data, output)
    
    logger.info("Synthetic data generation completed successfully.")

if __name__ == "__main__":
    main()
