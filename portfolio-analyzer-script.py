"""
Complete Portfolio Analysis Tool
Features: Performance Analysis, Seasonal Patterns, Staking Returns, and Tax Impact
"""

# Remove any previous imports and replace with these
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import calendar
import os
import time
import webbrowser
import warnings
import traceback
import logging
import sys

# Add these debug prints
# print("Imports completed successfully")
warnings.filterwarnings('ignore')
# print("Warnings filtered")

def debug_step(func):
    """Decorator to add debugging to methods"""
    def wrapper(*args, **kwargs):
        try:
            logging.debug(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            logging.debug(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    return wrapper

class Config:
    """Configuration settings for the portfolio analyzer"""
    def __init__(self):
        self.INVESTMENT = {
            'biweekly_amount': 300,
            'stock_allocation': 0.5,
            'crypto_allocation': 0.5
        }
        
        self.ASSETS = {
            'stocks': ['NVDA', 'AMZN', 'GOOGL', 'AAPL', 'PLTR', 'TSM', 'MSFT'],
            'crypto': ['DOGE', 'BNB', 'ETH', 'BTC', 'ADA', 'SOL', 'ATOM', 'XRP']
        }
        
        self.STAKING = {
            'ETH': 0.04,   # 4% APY
            'SOL': 0.07,   # 7% APY
            'ADA': 0.05,   # 5% APY
            'ATOM': 0.06,  # 6% APY
            'BNB': 0.05    # 5% APY
        }
        
        self.TAX = {
            'federal_rate': 0.25,     # 25% federal tax
            'indiana_rate': 0.0323,   # 3.23% Indiana state tax
            'long_term_rate': 0.15    # 15% long-term capital gains
        }
        
        self.DATES = {
            'start': '2020-01-01',
            'end': datetime.now().strftime('%Y-%m-%d')
        }

class PortfolioAnalyzer:
    def __init__(self):
        try:
            logging.info("Initializing PortfolioAnalyzer...")
            self.config = Config()
            self.output_folder = self.setup_output_folder()
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.data = {}
            self.metrics = {}
            self.seasonal_patterns = {}
            self.staking_returns = {}
            self.tax_impact = {}
            logging.debug("PortfolioAnalyzer initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing PortfolioAnalyzer: {str(e)}")
            raise

    def setup_output_folder(self):
        """Create and return output folder path"""
        folder = os.path.expanduser(os.path.join('~', 'Documents', 'PortfolioAnalysis'))
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

    def wait_for_user(self, message="\nPress Enter to continue..."):
        """Wait for user input"""
        try:
            input(message)
        except Exception:
            time.sleep(5)

    def handle_error(self, error):
        """Handle errors gracefully"""
        print(f"\nâŒ An error occurred: {str(error)}")
        print("\nError details:")
        traceback.print_exc()
        self.wait_for_user("\nPress Enter to continue...")

    def fetch_data(self):
        """Fetch all market data"""
        print("\n### Fetching Market Data:")
        print("=" * 50)

        # Fetch stock data
        print("\n>> Fetching Stock Data:")
        print("-" * 50)
        for stock in self.config.ASSETS['stocks']:
            print(f"Fetching {stock}...", end='', flush=True)
            try:
                ticker = yf.Ticker(stock)
                data = ticker.history(
                    start=self.config.DATES['start'],
                    end=self.config.DATES['end']
                )
                if len(data) > 0:
                    self.data[stock] = data
                    print(f" [OK] (${data['Close'].iloc[-1]:.2f})")
                else:
                    print(" [NO DATA]")
            except Exception as e:
                print(f" [ERROR]: {str(e)}")
            time.sleep(1)
        
        # Fetch crypto data
        print("\nFetching Crypto Data:")
        print("-" * 50)
        for crypto in self.config.ASSETS['crypto']:
            print(f"Fetching {crypto}...", end='', flush=True)
            try:
                ticker = yf.Ticker(f"{crypto}-USD")
                data = ticker.history(
                    start=self.config.DATES['start'],
                    end=self.config.DATES['end']
                )
                if len(data) > 0:
                    self.data[crypto] = data
                    print(f" (${data['Close'].iloc[-1]:.2f})")
                else:
                    print(" No data")
            except Exception as e:
                print(f" Error: {str(e)}")
            time.sleep(1)

    # @debug_step
    def calculate_metrics(self):
        """Calculate performance metrics"""
        print("\nCalculating Performance Metrics...")
        
        for asset, data in self.data.items():
            try:
                prices = data['Close']
                returns = prices.pct_change()
                
                # Basic metrics
                total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                annual_return = (1 + total_return) ** (252/len(returns)) - 1
                volatility = returns.std() * np.sqrt(252)
                sharpe = (annual_return - self.config.TAX['federal_rate']) / volatility if volatility != 0 else 0
                
                # Calculate drawdown
                rolling_max = prices.expanding().max()
                drawdown = (prices - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                self.metrics[asset] = {
                    'current_price': prices.iloc[-1],
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'price_history': prices
                }
                
            except Exception as e:
                print(f"Error calculating metrics for {asset}: {str(e)}")

    # @debug_step

    @debug_step
    def analyze_seasonal_patterns(self):
        """Calculate seasonal patterns"""
        print("\nAnalyzing Seasonal Patterns...")
        
        for asset, data in self.data.items():
            try:
                prices = data['Close']
                returns = prices.pct_change()
                dates = prices.index
                
                print(f"Analyzing {asset}...")
                print(f"Prices: {prices}")
                print(f"Returns: {returns}")
                print(f"Dates: {dates}")
                
                # Monthly analysis
                monthly_returns = returns.groupby(dates.month).mean()
                print(f"Monthly Returns: {monthly_returns}")
                
                # Quarterly analysis
                quarterly_returns = returns.groupby(dates.quarter).mean()
                print(f"Quarterly Returns: {quarterly_returns}")
                
                # Holiday effects
                holidays = {
                    'Christmas': (12, [20, 31]),
                    'New_Year': (1, [1, 10]),
                    'Independence_Day': (7, [1, 10]),
                    'Thanksgiving': (11, [20, 30])
                }
                
                holiday_returns = {}
                for holiday, (month, days) in holidays.items():
                    mask = (dates.month == month) & (dates.day.isin(range(days[0], days[1]+1)))
                    holiday_returns[holiday] = returns[mask].mean()
                
                print(f"Holiday Returns: {holiday_returns}")
                
                self.seasonal_patterns[asset] = {
                    'monthly': dict(monthly_returns),
                    'quarterly': dict(quarterly_returns),
                    'holidays': holiday_returns,
                    'best_month': monthly_returns.idxmax(),
                    'worst_month': monthly_returns.idxmin()
                }
                
            except Exception as e:
                print(f"Error analyzing patterns for {asset}: {str(e)}")

    # @debug_step
    def calculate_staking_returns(self):
        """Calculate staking returns for eligible cryptocurrencies"""
        print("\nCalculating Staking Returns...")
        
        for crypto, staking_rate in self.config.STAKING.items():
            if crypto in self.data:
                try:
                    prices = self.data[crypto]['Close']
                    days_held = (prices.index[-1] - prices.index[0]).days
                    
                    # Calculate daily staking return (compounded)
                    daily_rate = (1 + staking_rate) ** (1/365) - 1
                    
                    # Calculate investment amount
                    initial_investment = (
                        self.config.INVESTMENT['biweekly_amount'] * 
                        self.config.INVESTMENT['crypto_allocation'] / 
                        len(self.config.ASSETS['crypto'])
                    )
                    
                    # Calculate staking returns
                    staking_value = initial_investment * ((1 + daily_rate) ** days_held)
                    staking_return = staking_value - initial_investment
                    
                    self.staking_returns[crypto] = {
                        'apy': staking_rate,
                        'days_held': days_held,
                        'total_return': staking_return,
                        'annual_return': staking_rate,
                        'total_value': staking_value
                    }
                    
                except Exception as e:
                    print(f"Error calculating staking for {crypto}: {str(e)}")
                    
    # @debug_step
    def calculate_tax_impact(self):
        """Calculate tax implications"""
        print("\nCalculating Tax Impact...")
        
        for asset, asset_metrics in self.metrics.items():
            try:
                gains = asset_metrics['total_return'] * self.config.INVESTMENT['biweekly_amount']
                
                # Calculate taxes
                federal_tax = max(0, gains * self.config.TAX['federal_rate'])
                state_tax = max(0, gains * self.config.TAX['indiana_rate'])
                
                # Calculate after-tax return
                after_tax_return = asset_metrics['total_return'] - (
                    (federal_tax + state_tax) / self.config.INVESTMENT['biweekly_amount']
                )
                
                self.tax_impact[asset] = {
                    'federal_tax': federal_tax,
                    'state_tax': state_tax,
                    'after_tax_return': after_tax_return
                }
                
            except Exception as e:
                print(f"Error calculating tax impact for {asset}: {str(e)}")

    def generate_reports(self):
        """Generate all reports and visualizations"""
        print("\nGenerating Reports and Visualizations...")
        
        try:
            # Create report filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_report = os.path.join(self.output_folder, f'portfolio_report_{timestamp}.html')
            csv_report = os.path.join(self.output_folder, f'portfolio_data_{timestamp}.csv')
            charts_report = os.path.join(self.output_folder, f'portfolio_charts_{timestamp}.html')
            
            # Generate reports
            self.generate_html_report(html_report)
            self.generate_csv_report(csv_report)
            self.generate_interactive_charts(charts_report)
            
            # Open reports in browser
            print("\nOpening reports in browser...")
            webbrowser.open(f'file://{os.path.realpath(html_report)}')
            webbrowser.open(f'file://{os.path.realpath(charts_report)}')
            
        except Exception as e:
            print(f"Error generating reports: {str(e)}")
            traceback.print_exc()

    def generate_html_report(self, filename):
        """Generate detailed HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Portfolio Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 8px 8px 0 0;
                    margin-bottom: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                    background-color: white;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                    font-weight: 600;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: white;
                    border-radius: 8px;
                }}
                .positive {{
                    color: #28a745;
                    font-weight: 500;
                }}
                .negative {{
                    color: #dc3545;
                    font-weight: 500;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Portfolio Analysis Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
        """
        
        # Add performance summary
        html_content += self.generate_performance_summary_html()
        
        # Add seasonal analysis
        html_content += self.generate_seasonal_analysis_html()
        
        # Add staking returns
        html_content += self.generate_staking_returns_html()
        
        # Add tax impact
        html_content += self.generate_tax_impact_html()
        
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML Report saved to: {filename}")


    def generate_interactive_charts(self, filename):
        """Generate interactive Plotly charts"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    'Asset Price History',
                    'Total Returns Comparison',
                    'Monthly Performance Patterns',
                    'Risk Metrics'
                ),
                vertical_spacing=0.1,
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Price history chart
            for asset in self.metrics:
                fig.add_trace(
                    go.Scatter(
                        x=self.metrics[asset]['price_history'].index,
                        y=self.metrics[asset]['price_history'].values,
                        name=f"{asset} Price",
                        mode='lines'
                    ),
                    row=1, col=1
                )
            
            # Returns comparison
            returns_data = []
            for asset in self.metrics:
                returns_data.append(
                    go.Bar(
                        name=asset,
                        x=['Total Return'],
                        y=[self.metrics[asset]['total_return'] * 100]
                    )
                )
            for trace in returns_data:
                fig.add_trace(trace, row=2, col=1)
            
            # Monthly patterns
            for asset in self.seasonal_patterns:
                monthly_returns = self.seasonal_patterns[asset]['monthly']
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, 13)),
                        y=[monthly_returns.get(m, 0) * 100 for m in range(1, 13)],
                        name=f"{asset} Monthly",
                        mode='lines+markers'
                    ),
                    row=3, col=1
                )
            
            # Risk metrics
            risk_data = go.Bar(
                x=list(self.metrics.keys()),
                y=[m['volatility'] * 100 for m in self.metrics.values()],
                name='Volatility (%)'
            )
            fig.add_trace(risk_data, row=4, col=1)
            
            # Update layout
            fig.update_layout(
                height=1600,
                showlegend=True,
                title_text="Portfolio Analysis Dashboard",
                template="plotly_white"
            )
            
            # Update axes
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Return (%)", row=2, col=1)
            fig.update_yaxes(title_text="Monthly Return (%)", row=3, col=1)
            fig.update_yaxes(title_text="Volatility (%)", row=4, col=1)
            
            # Save chart
            fig.write_html(filename)
            print(f"Interactive charts saved to: {filename}")
            
        except Exception as e:
            print(f"Error generating charts: {str(e)}")
            traceback.print_exc()

    def generate_csv_report(self, filename):
        """Generate detailed CSV report"""
        try:
            data = []
            
            # Compile metrics
            for asset in self.metrics:
                row = {
                    'Asset': asset,
                    'Current_Price': self.metrics[asset]['current_price'],
                    'Total_Return': self.metrics[asset]['total_return'],
                    'Annual_Return': self.metrics[asset]['annual_return'],
                    'Volatility': self.metrics[asset]['volatility'],
                    'Sharpe_Ratio': self.metrics[asset]['sharpe_ratio'],
                    'Max_Drawdown': self.metrics[asset]['max_drawdown']
                }
                
                # Add seasonal patterns
                if asset in self.seasonal_patterns:
                    row.update({
                        'Best_Month': self.seasonal_patterns[asset]['best_month'],
                        'Worst_Month': self.seasonal_patterns[asset]['worst_month']
                    })
                
                # Add staking returns
                if asset in self.staking_returns:
                    row.update({
                        'Staking_APY': self.staking_returns[asset]['apy'],
                        'Staking_Return': self.staking_returns[asset]['total_return']
                    })
                
                # Add tax impact
                if asset in self.tax_impact:
                    row.update({
                        'Federal_Tax': self.tax_impact[asset]['federal_tax'],
                        'State_Tax': self.tax_impact[asset]['state_tax'],
                        'After_Tax_Return': self.tax_impact[asset]['after_tax_return']
                    })
                
                data.append(row)
            
            # Create and save DataFrame
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"CSV Report saved to: {filename}")
            
        except Exception as e:
            print(f"Error generating CSV: {str(e)}")
            traceback.print_exc()

    def generate_performance_summary_html(self):
        """Generate HTML for performance summary"""
        html = """
            <div class="section">
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Asset</th>
                        <th>Current Price</th>
                        <th>Total Return</th>
                        <th>Annual Return</th>
                        <th>Volatility</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                    </tr>
        """
        
        for asset, metrics in self.metrics.items():
            html += f"""
                <tr>
                    <td><strong>{asset}</strong></td>
                    <td>${metrics['current_price']:,.2f}</td>
                    <td class="{'positive' if metrics['total_return'] > 0 else 'negative'}">
                        {metrics['total_return']:.2%}
                    </td>
                    <td class="{'positive' if metrics['annual_return'] > 0 else 'negative'}">
                        {metrics['annual_return']:.2%}
                    </td>
                    <td>{metrics['volatility']:.2%}</td>
                    <td>{metrics['sharpe_ratio']:.2f}</td>
                    <td class="negative">{metrics['max_drawdown']:.2%}</td>
                </tr>
            """
        
        html += "</table></div>"
        return html

    def generate_seasonal_analysis_html(self):
        """Generate HTML for seasonal analysis"""
        html = """
            <div class="section">
                <h2>Seasonal Analysis</h2>
                <table>
                    <tr>
                        <th>Asset</th>
                        <th>Best Month</th>
                        <th>Worst Month</th>
                        <th>Best Quarter</th>
                        <th>Holiday Effect</th>
                    </tr>
        """
        
        for asset, patterns in self.seasonal_patterns.items():
            best_month = calendar.month_name[patterns['best_month']]
            worst_month = calendar.month_name[patterns['worst_month']]
            
            # Find best quarter
            best_quarter = max(patterns['quarterly'].items(), key=lambda x: x[1])[0]
            
            # Get best holiday effect
            best_holiday = max(patterns['holidays'].items(), key=lambda x: x[1])[0]
            
            html += f"""
                <tr>
                    <td><strong>{asset}</strong></td>
                    <td>{best_month}</td>
                    <td>{worst_month}</td>
                    <td>Q{best_quarter}</td>
                    <td>{best_holiday.replace('_', ' ')}</td>
                </tr>
            """
        
        html += "</table></div>"
        return html

    def generate_staking_returns_html(self):
        """Generate HTML for staking returns"""
        if not self.staking_returns:
            return ""
            
        html = """
            <div class="section">
                <h2>Staking Returns</h2>
                <table>
                    <tr>
                        <th>Asset</th>
                        <th>APY</th>
                        <th>Days Held</th>
                        <th>Total Return</th>
                        <th>Total Value</th>
                    </tr>
        """
        
        for crypto, returns in self.staking_returns.items():
            html += f"""
                <tr>
                    <td><strong>{crypto}</strong></td>
                    <td>{returns['apy']:.2%}</td>
                    <td>{returns['days_held']}</td>
                    <td class="positive">${returns['total_return']:,.2f}</td>
                    <td>${returns['total_value']:,.2f}</td>
                </tr>
            """
        
        html += "</table></div>"
        return html

    def generate_tax_impact_html(self):
        """Generate HTML for tax impact"""
        html = """
            <div class="section">
                <h2>Tax Impact</h2>
                <table>
                    <tr>
                        <th>Asset</th>
                        <th>Federal Tax</th>
                        <th>State Tax</th>
                        <th>After-Tax Return</th>
                    </tr>
        """
        
        for asset, tax in self.tax_impact.items():
            html += f"""
                <tr>
                    <td><strong>{asset}</strong></td>
                    <td>${tax['federal_tax']:,.2f}</td>
                    <td>${tax['state_tax']:,.2f}</td>
                    <td class="{'positive' if tax['after_tax_return'] > 0 else 'negative'}">
                        {tax['after_tax_return']:.2%}
                    </td>
                </tr>
            """
        
        html += "</table></div>"
        return html

    def run_analysis(self):
        """Run the complete portfolio analysis"""
        try:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Print welcome message
            print("Complete Portfolio Analysis Tool")
            print("=" * 50)
            print("\nConfiguration:")
            print(f"Start Date: {self.config.DATES['start']}")
            print(f"End Date: {self.config.DATES['end']}")
            print(f"Biweekly Investment: ${self.config.INVESTMENT['biweekly_amount']}")
            print("\nStocks:", ', '.join(self.config.ASSETS['stocks']))
            print("Cryptocurrencies:", ', '.join(self.config.ASSETS['crypto']))
            
            # Fetch market data
            self.fetch_data()
            
            # Calculate metrics
            if self.data:
                self.calculate_metrics()
                self.analyze_seasonal_patterns()
                self.calculate_staking_returns()
                self.calculate_tax_impact()
                
                # Generate reports
                self.generate_reports()
                
                print("\n[SUCCESS] Analysis complete! Reports have been generated and opened in your browser.")
            else:
                print("\n[ERROR] No data was fetched. Unable to perform analysis.")
            
            print("\nPress Enter to exit...")
            input()
            
        except Exception as error:
            self.handle_error(error)
def main():
    """Main execution function with enhanced error handling"""
    print("Starting main function...")  # Debug print
    try:
        # Create logs directory in the same folder as the script
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print(f"Created log directory at: {log_dir}")  # Debug print
        
        log_file = os.path.join(log_dir, 'portfolio_analysis.log')
        print(f"Log file will be created at: {log_file}")  # Debug print
        
        logging.basicConfig(
            level=logging.INFO,  # Change this from logging.DEBUG to logging.INFO or logging.WARNING
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        print("Logging configured, starting analysis...")  # Debug print
        logging.info("Starting Portfolio Analysis Tool...")
        analyzer = PortfolioAnalyzer()
        analyzer.run_analysis()
    except Exception as e:
        print(f"\nError in main: {str(e)}")  # Debug print
        logging.critical(f"Fatal error: {str(e)}")
        logging.critical(traceback.format_exc())
        print("\n[X] Fatal error occurred.")
        print("Check portfolio_analysis.log for details.")
    finally:
        print("Press Enter to exit...")  # Debug print
        input()

if __name__ == "__main__":
    print("Script started...")  # Debug print
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        input("Press Enter to exit...")
