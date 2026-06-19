import yfinance as yf
import pandas as pd


class AssetDataIngestion:

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def yf_download(self):

        # List of tickers needed for the analysis

        equity_tickers = ['^GSPC', 'FEZ']
        energy_tickers = [ 'CL=F', 'NG=F']
        fx_tickers = ['EURUSD=X', 'USDCHF=X']
        vix_ticker = '^VIX'
        all_tickers = equity_tickers + energy_tickers + fx_tickers + [vix_ticker]

        # ^GSPC = S&P500, FEZ = stoxx50, CL=F Crude oil WTI, BZ=F Crude oil Brent, NG=F Natural gas
         
        df_list = []

        for el in all_tickers:
            df = yf.download(el, start=self.start_date, end=self.end_date)['Close']
            df.name= el
            df_list.append(df)

        df_yfinance_data = pd.concat(df_list, axis = 1)

        return df_yfinance_data

    def eia_download(self):

        # Daily prices for Brent Crude Oil since 1987
        df_brent_eia = pd.read_excel("https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls", sheet_name="Data 1", skiprows=2)

        df_brent_eia['Date'] = pd.to_datetime(df_brent_eia['Date'])
        df_brent_eia.set_index('Date', inplace=True)

        df_brent_eia = df_brent_eia.loc[self.start_date:self.end_date]
        df_brent_eia.columns = ['Brent_EIA']
       

        return df_brent_eia

    def get_combined_data(self, drop_na=True):

        df_yf = self.yf_download()
        df_brent = self.eia_download()
        df_raw = pd.concat([df_yf, df_brent], axis=1)
        if drop_na:
            df_raw = df_raw.dropna()
        return df_raw




def main():
    ingest = AssetDataIngestion("2004-01-05", "2025-12-31")
    df_raw = ingest.get_combined_data()
    df_raw.to_csv("df_historical_prices.csv")

if __name__ == "__main__":
    main()

    