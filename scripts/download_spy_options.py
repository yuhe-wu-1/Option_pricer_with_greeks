# Downloads a public SPY options history (parquet) used for early exploration. Not needed for the main analysis.
import urllib.request

ticker = "spy"  # use lowercase
urllib.request.urlretrieve(
    f"https://static.philippdubach.com/data/options/{ticker}/options.parquet",
    f"{ticker}_options.parquet"
)