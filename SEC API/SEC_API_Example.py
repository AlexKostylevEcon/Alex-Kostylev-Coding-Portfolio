# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:24:51 2023

@author: alexs
"""

import json
from secedgar.cik_lookup import get_cik_map
import urllib.request, json 
import requests
import pandas as pd 
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
import html2text
import urllib.request
from urllib.request import Request, urlopen  

header = {
  "User-Agent": "Some Email"#, # remaining fields are optional
#    "Accept-Encoding": "gzip, deflate",
#    "Host": "data.sec.gov"
}

tickers = pd.DataFrame.from_dict(requests.get("https://www.sec.gov/files/company_tickers.json", headers=header).json(), orient='index')

CIK=tickers["cik_str"][0]

url = f"https://data.sec.gov/submissions/CIK{str(CIK).zfill(10)}.json"
url

company_filings = requests.get(url, headers=header).json()
company_filings.keys()
company_filings["addresses"]
company_filings["filings"]["recent"].keys()
company_filings_df = pd.DataFrame(company_filings["filings"]["recent"])
company_filings_df

access_numbers = company_filings_df[company_filings_df.form == "10-K"].accessionNumber.values
access_number=access_numbers[0].replace("-", "")
file_name = company_filings_df[company_filings_df.form == "10-K"].primaryDocument.values[0]
url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{access_number}/{file_name}"
url

# TXT file is  f"https://www.sec.gov/Archives/edgar/data/{CIK}/{access_number}/{access_number[0:10]}-{access_number[10:12]}-{access_number[12:]}.txt"

#url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{access_number}/{access_number[0:10]}-{access_number[10:12]}-{access_number[12:]}.txt"

req_content = requests.get(url, headers=header).content
rendered_content = html2text.html2text(req_content.decode("utf-8"))
file=open("test.txt", "w", encoding="utf-8")
file.write(rendered_content)
file.close()

with open( 'output.html', 'wb' ) as download:
    download.write(req_content)

# Downloading texts

dl = Downloader("MyCompanyName", "Some Email", "Some Path")
dl.get("10-K", "MSFT", download_details=False)

#########################################

# Company facts
# This is there I will take all information to create report tables

url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(CIK).zfill(10)}.json"
url
company_facts = requests.get(url, headers=header).json()
curr_assets_df = pd.DataFrame(company_facts["facts"]["us-gaap"]["AssetsCurrent"]["units"]["USD"])
curr_assets_df

curr_assets_df[curr_assets_df.frame.notna()]

import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
pd.options.plotting.backend = "plotly" 

pio.renderers.default='browser'

curr_assets_df.plot(x="end", y="val", 
                    title=f"{company_filings['name']}, {CIK}: Current Assets",
                   labels= {
                       "val": "Value ($)",
                       "end": "Quarter End"
                   })

# It will be easier to manipulate metrics at report level rather than connect reports from metric dataframes
# Single metric for all companies

fact = "AssetsCurrent"
year = 2023
quarter = "Q3I"

url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{fact}/USD/CY{year}{quarter}.json"
url

curr_assets_dict = requests.get(url, headers=header).json()
curr_assets_dict.keys()
curr_assets_df = pd.DataFrame(curr_assets_dict["data"])
curr_assets_df.sort_values("val", ascending=False)

facts_names=list(company_facts["facts"]["us-gaap"].keys())

# Can get annual report values form here, althout it will probably take a very long time

url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{str(CIK).zfill(10)}/us-gaap/AccountsPayableCurrent.json"
url
company_concept = requests.get(url, headers=header).json()

# Getting ticker prices to generate financial performance variable

import yfinance as yf
ticker = yf.Ticker(tickers["ticker"][0]).info
market_price = ticker['regularMarketPrice']
previous_close_price = ticker['regularMarketPreviousClose']

start_date = '2020-01-01'
end_date = '2022-01-01'

ticker=tickers["ticker"][0]
 
# Get the data
data = yf.download(ticker, start_date, end_date)

period_return=data["Close"][-1]/data["Close"][0]-1
period_return

# Connecting everything together

# Generate dataframe with companies in rows and financial variables from annual report in columns
# These will be independent variables

# Get the date of the report and the date 3 months, 6 months and a year from the date of report publication.
# Get ticker closing prices for all dates and generate performance variables. 
# These will be dependent variables

# For some reason there are almost no yearly or quarterly frames. Only Instantaneous ones.
"""
fact = "AccountsPayableCurrent"
year = 2022
quarter = "Q2I"
#quarter = ""

url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{fact}/USD/CY{year}{quarter}.json"
url

curr_assets_dict = requests.get(url, headers=header).json()
curr_assets_dict.keys()
curr_assets_df = pd.DataFrame(curr_assets_dict["data"])
curr_assets_df.sort_values("val", ascending=False)
"""

url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(CIK).zfill(10)}.json"
url
company_facts = requests.get(url, headers=header).json()
curr_assets_df = pd.DataFrame(company_facts["facts"]["us-gaap"]["AssetsCurrent"]["units"]["USD"])
curr_assets_df


company_facts["facts"]["us-gaap"].keys()

# First, create a dataframe for a company with all metrics from the latest annual report

curr_assets_df = pd.DataFrame(company_facts["facts"]["us-gaap"]["AccountsPayableCurrent"]["units"]["USD"])
curr_assets_df=curr_assets_df.loc[curr_assets_df["form"]=="10-K"]

curr_assets_df=curr_assets_df.drop_duplicates(subset=['accn', 'fy'], keep='last')

#value in the last year and submissing ID 
last_value=curr_assets_df.iloc[-1]["val"]
last_10K_accn=curr_assets_df.iloc[-1]["accn"]

# accn will allow to connect all metrics existing in a single report
# Now will need to loop through all dataframes for individual metrics and create a dataframe with all metrics for a given company and a given report
# This will then become a building block of the dataframe of dataframe for all companies with the last annual report
# Then, additional reports can be added as duplicate variables with _t1 _2 etc subscript
# Then, these will become independent variables in prediction exercise, where stock performance is a dependent variable


import xml.etree.ElementTree as ET

file_name = company_filings_df[company_filings_df.form == "10-K"].primaryDocument.values[0]


url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{access_number}/{file_name}"
xmlfile=requests.get(url, headers=header).content

req = urllib.request.Request(url, headers=header)
response = urllib.request.urlopen(req)

#.content.decode("utf-8")

with urllib.request.urlopen(req) as url:
    html = url.read()
soup = BeautifulSoup(html, "html.parser")
for table in soup.find_all("table"):
    table.decompose()
for script in soup(["script", "style"]):
    script.extract()  
text = soup.get_text()
print (text)
