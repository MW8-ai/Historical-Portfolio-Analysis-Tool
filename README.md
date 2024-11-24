Version 1 Beta

# Portfolio Analyzer

This Python script analyzes historical stock and cryptocurrency prices to provide insights and assist in making informed investment decisions using the Dollar-Cost Averaging (DCA) strategy. The script retrieves data from the Yahoo Finance API and generates analysis charts and a CSV export.

## Features

- Fetches historical price data for stocks and cryptocurrencies
- Performs analysis on the retrieved data
- Generates interactive charts for visualization
- Exports the analysis results to a CSV file
- Helps in understanding the potential outcomes of different investment strategies

## Getting Started

### Prerequisites

- Python 3.x installed on your system

### Installation

1. Clone the repository or download the script file.

2. Install the required dependencies by running the following command:

   ```
   pip install pandas numpy yfinance plotly
   ```

   or

   ```
   pip3 install pandas numpy yfinance plotly
   ```

### Usage

To run the script, use the following command:

python portfolio_analyzer.py
or
python3 portfolio_analyzer.py


The script will fetch the necessary data, perform the analysis, and generate the output files.

## Configuration

You can modify the script's configuration by editing the following variables:

- `INVESTMENT`: Adjust the investment amount and allocation percentages.
- `ASSETS`: Update the list of stocks and cryptocurrencies to analyze.
- `DATES`: Set the start and end dates for the analysis.

## Disclaimer

Please note that this script is provided for informational purposes only. It is not intended as financial advice. Always conduct your own research and consult with a qualified financial advisor before making any investment decisions.

## Future Enhancements

- Improve the accuracy of the DCA calculation
- Add more analysis metrics and visualizations
- Implement a user-friendly interface
- Optimize the script's performance

## Contributing

Contributions to the project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize the README.md file further based on your specific requirements and project details.
