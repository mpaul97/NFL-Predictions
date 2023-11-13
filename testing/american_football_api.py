import pandas as pd
import numpy as np
import os
import requests

url = "https://api-american-football.p.rapidapi.com/games"

querystring = {"date":"2022-09-30"}

headers = {
	"X-RapidAPI-Key": "9fa9382f92msh7a31d64d5592c88p168c3djsn1401d40dbe5c",
	"X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())