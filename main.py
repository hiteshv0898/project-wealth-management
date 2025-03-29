from src.utils.logger import log_info, log_error
from src.visualization.plotter import plot_predictions
import numpy as np

def main():
    try:
        log_info("Starting Wealth Management Agent")

        actual = np.random.uniform(200, 300, 10)
        predicted = np.random.uniform(200, 300, 10)
        plot_predictions(actual, predicted)

        log_info("Plot generated successfully")
    except Exception as e:
        log_error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()