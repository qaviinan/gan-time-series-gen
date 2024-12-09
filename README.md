# High-Frequency Crypto GAN CLI Tool

![Crypto GAN](https://example.com/crypto-gan-banner.png)

Welcome to **High-Frequency Crypto GAN CLI Tool**, a powerful command-line application designed to generate high-frequency synthetic cryptocurrency time-series data using a Generative Adversarial Network (GAN) approach.

## 🚀 Features

- **Synthetic Data Generation:** Produce high-quality, realistic cryptocurrency price data (e.g., Bitcoin) based on historical data.
- **Flexible Configuration:** Customize model parameters such as iterations, batch size, hidden dimensions, and more.
- **Easy Installation:** Install via `pip` or use the Dockerized version for seamless setup.
- **Sample Data Included:** Start generating immediately with provided sample Bitcoin price data.
- **Extensible Design:** Built with modularity in mind, allowing for easy enhancements and integrations.
- **Comprehensive CLI:** User-friendly command-line interface powered by `click` for straightforward operations.
- **Visualization Support:** Optionally visualize the generated data compared to real data.

## 📦 Installation

Choose the installation method that best suits your workflow. You can install the tool using `pip` or leverage Docker for containerized deployment.

### 1. Installation via `pip`

#### Prerequisites

- **Python 3.6+**
- **pip** package manager

#### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/high_freq_crypto_gan.git
   cd high_freq_crypto_gan
   ```

2. **Install Dependencies**

   Ensure you have the required dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the CLI Tool**

   ```bash
   pip install .
   ```

   *Note: Depending on your system, you might need to use `pip3` instead of `pip`.*

### 2. Installation via Docker

#### Prerequisites

- **Docker** installed on your system.

#### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/high_freq_crypto_gan.git
   cd high_freq_crypto_gan
   ```

2. **Build the Docker Image**

   ```bash
   docker build -t high_freq_crypto_gan .
   ```

3. **Run the Docker Container**

   ```bash
   docker run --rm \
       -v $(pwd)/sample_data:/app/sample_data \
       -v $(pwd)/generated_data:/app/generated_data \
       high_freq_crypto_gan \
       --input sample_data/bitcoin_prices.csv \
       --output generated_data/synthetic_prices.csv \
       --iterations 20000 \
       --batch_size 128
   ```

   *Explanation:*
   - `-v $(pwd)/sample_data:/app/sample_data`: Mounts the `sample_data` directory.
   - `-v $(pwd)/generated_data:/app/generated_data`: Mounts the `generated_data` directory to store outputs.
   - Replace the options after the image name with your desired parameters.

## 🛠 Usage

Once installed, you can use the `crypto-gan` command to generate synthetic crypto price data. Below are detailed instructions on how to use the CLI tool, including available options and example commands.

### Basic Command Structure

```bash
crypto-gan [OPTIONS]
```

### Available Options

| Option           | Short Flag | Type    | Default                          | Description                                               |
|------------------|------------|---------|----------------------------------|-----------------------------------------------------------|
| `--input`        | `-i`       | `str`   | `sample_data/bitcoin_prices.csv`  | Path to the input CSV file containing crypto prices.      |
| `--output`       | `-o`       | `str`   | `generated_data/synthetic_prices.csv` | Path to save the generated synthetic data.            |
| `--iterations`   | `-n`       | `int`   | `20000`                           | Number of training iterations.                            |
| `--batch_size`   | `-b`       | `int`   | `128`                             | Batch size for training.                                  |
| `--z_dim`        |            | `int`   | `24`                              | Dimension of the random noise vector.                     |
| `--hidden_dim`   |            | `int`   | `24`                              | Hidden dimension size for RNN cells.                      |
| `--num_layers`   |            | `int`   | `3`                               | Number of layers in RNN and transformer.                  |
| `--d_model`      |            | `int`   | `24`                              | Model dimension for the transformer.                      |
| `--num_heads`    |            | `int`   | `2`                               | Number of attention heads in the transformer.              |
| `--dff`          |            | `int`   | `128`                             | Feed-forward network dimension in the transformer.         |
| `--visualize`    |            | Flag    | `False`                           | Visualize the generated data compared to the original data.|

### Example Commands

#### 1. Generate Synthetic Data with Default Parameters

```bash
crypto-gan
```

*This command uses the default input and output paths along with default training parameters.*

#### 2. Specify Custom Input and Output Paths

```bash
crypto-gan --input data/real_crypto_prices.csv --output data/synthetic_crypto_prices.csv
```

#### 3. Customize Training Parameters

```bash
crypto-gan \
    --iterations 50000 \
    --batch_size 256 \
    --z_dim 32 \
    --hidden_dim 64 \
    --num_layers 4 \
    --d_model 64 \
    --num_heads 4 \
    --dff 256
```

#### 4. Enable Visualization of Results

```bash
crypto-gan --visualize
```

*This command will generate synthetic data and display a plot comparing the original and synthetic data.*

### Complete Example

```bash
crypto-gan \
    --input sample_data/bitcoin_prices.csv \
    --output generated_data/synthetic_prices.csv \
    --iterations 30000 \
    --batch_size 128 \
    --z_dim 24 \
    --hidden_dim 24 \
    --num_layers 3 \
    --d_model 24 \
    --num_heads 2 \
    --dff 128 \
    --visualize
```

*This command trains the GAN for 30,000 iterations with a batch size of 128, generates synthetic data, saves it to `generated_data/synthetic_prices.csv`, and visualizes the results.*

## 📁 Project Structure

```
high_freq_crypto_gan/
├── models/
│   ├── gan.py
│   └── __init__.py
├── utils/
│   ├── data_processing.py
│   ├── helpers.py
│   └── __init__.py
├── sample_data/
│   └── bitcoin_prices.csv
├── cli/
│   ├── __init__.py
│   └── main.py
├── tests/
│   └── test_cli.py
├── generated_data/
│   └── synthetic_prices.csv
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

- **models/**: Contains the GAN model implementation.
- **utils/**: Includes utility functions for data processing and model helpers.
- **sample_data/**: Provides sample Bitcoin price data for immediate testing.
- **cli/**: Houses the command-line interface implementation using `click`.
- **tests/**: Contains unit tests to ensure the CLI tool functions correctly.
- **generated_data/**: Destination for the generated synthetic data.
- **requirements.txt**: Lists all Python dependencies.
- **setup.py**: Configuration file for packaging and distribution.
- **Dockerfile**: Defines the Docker image for containerized deployment.
- **README.md**: Project documentation.

## 🧪 Testing

Ensure that the CLI tool works as expected by running the provided unit tests.

### Running Tests

1. **Navigate to the Project Root**

   ```bash
   cd high_freq_crypto_gan
   ```

2. **Execute the Tests**

   ```bash
   python -m unittest discover tests
   ```

   *This command discovers and runs all tests in the `tests/` directory.*

### Example Test: `tests/test_cli.py`

```python
# tests/test_cli.py

import unittest
import subprocess
import os

class TestCryptoGANCLI(unittest.TestCase):
    def test_cli_execution(self):
        input_path = "sample_data/bitcoin_prices.csv"
        output_path = "tests/generated_synthetic_prices.csv"
        cmd = [
            "crypto-gan",
            "--input", input_path,
            "--output", output_path,
            "--iterations", "1000",
            "--batch_size", "32"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertEqual(result.returncode, 0)
            self.assertTrue(os.path.exists(output_path))
            # Optionally, check if the output file has content
            with open(output_path, 'r') as f:
                content = f.read()
                self.assertTrue(len(content) > 0)
        except subprocess.CalledProcessError as e:
            self.fail(f"CLI execution failed: {e}")

if __name__ == "__main__":
    unittest.main()
```

*This test verifies that the CLI command runs successfully and generates the expected output file.*

## 🖥️ Development & Contribution

We welcome contributions from the community! To contribute:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

   *Provide a clear description of your changes and the problem they solve.*

---

### Coding Standards

- Follow [PEP 8](https://pep8.org/) style guidelines.
- Write clear and concise commit messages.
- Include tests for new features or bug fixes.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 📫 Contact

For questions, suggestions, or support, please contact:

- **Your Name** - [your.email@example.com](mailto:your.email@example.com)
- **GitHub:** [https://github.com/yourusername/high_freq_crypto_gan](https://github.com/yourusername/high_freq_crypto_gan)