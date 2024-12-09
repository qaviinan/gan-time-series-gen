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
