import requests
from bs4 import BeautifulSoup
import json

EVENTS = [
    ('50',"freestyle"),
    ('100',"freestyle"),
    ('200',"freestyle"),
    ('400',"freestyle"),
    ('800',"freestyle"),
    ('1500',"freestyle"),
    ('100',"backstroke"),
    ('200',"backstroke"),
    ('100',"breaststroke"),
    ('200',"breaststroke"),
    ('100',"butterfly"),
    ('200',"butterfly"),
    ('200',"medley"),
    ('400',"medley")
]

def parse_time(time):
    #https://stackoverflow.com/a/41252517
    return sum(x * float(t) for x, t in zip([1, 60, 3600], time.split(":")[::-1]))

def download_event(event):
    distance, stroke = event
    years = [x for x in range(1970,2022)]
    year_to_best_times = {}
    for year in years:
        url = f"https://api.fina.org/fina/rankings/swimming?gender=M&distance={distance}&stroke={stroke.upper()}&poolConfiguration=LCM&year={year}&startDate=&endDate=&timesMode=BEST_TIMES&regionId=&countryId=&pageSize=200"
        r = requests.get(url)
        data_dict = json.loads(r.text)
        best_times = [-1 * parse_time(x["time"]) for x in data_dict["swimmingWorldRankings"]]
        year_to_best_times[year] = best_times
    open(f"/tmp/records_{''.join(event)}.json","w").write(json.dumps(year_to_best_times))

if __name__ == "__main__":
    for event in EVENTS:
        print(event)
        download_event(event)
