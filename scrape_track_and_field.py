import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime

EVENT_TO_SIGN = {
    "100m": -1,
    "200m": -1,
    "400m": -1,
    "800m": -1,
    "1500m": -1,
    "5000m": -1,
    "10000m": -1,
    "marathon": -1,
    "110mh": -1,
    "400mh": -1,
    "3000mSC": -1,
    "high_jump": 1,
    "pole_vault": 1,
    "long_jump": 1,
    "triple_jump": 1,
    "shot_put": 1,
    "discus": 1,
    "hammer": 1,
    "javelin": 1,
    "decathlon": 1
}

def parse_time(time):
    time = time.replace("A","") #altitude
    time = time.replace("y","") #yards, converted to meters(!)
    time = time.replace("#","") #???
    time = time.replace("+","") #???
    time = time.replace("a","") #???
    time = time.replace("d","") #???
    time = time.replace("h","") #???
    time = time.replace("´","") #???
    time = time.replace("*","") #???
    #https://stackoverflow.com/a/41252517
    return sum(x * float(t) for x, t in zip([1, 60, 3600], time.split(":")[::-1]))

def get_world_url(event, year, page):
    #TODO: test reformatted urls
    root = "https://www.worldathletics.org/records/all-time-toplists"

    first_day = f"{year}-01-01"
    if year == datetime.today().year:
        #can't search for dates after today
        last_day = datetime.today().strftime('%Y-%m-%d')
    else:
        last_day = f"{year}-12-31"
    params = f"regionType=world&bestResultsOnly=true&page={page}&firstDay={first_day}&lastDay={last_day}"
    return {
        "100m": f"{root}/sprints/100-metres/outdoor/men/senior?{params}&timing=electronic&windReading=regular",
        "200m": f"{root}/sprints/200-metres/outdoor/men/senior?{params}&timing=electronic&windReading=regular",
        "400m": f"{root}/sprints/400-metres/outdoor/men/senior?{params}&timing=electronic",
        "800m": f"{root}/middlelong/800-metres/outdoor/men/senior?{params}&timing=electronic",
        "1500m": f"{root}/middlelong/1500-metres/outdoor/men/senior?{params}",
        "5000m": f"{root}/middlelong/5000-metres/outdoor/men/senior?{params}",
        "10000m": f"{root}/middlelong/10000-metres/outdoor/men/senior?{params}",
        "marathon": f"{root}/road-running/marathon/outdoor/men/senior?{params}&drop=regular&fiftyPercentRule=regular",
        "110mh": f"{root}/hurdles/110-metres-hurdles/outdoor/men/senior?{params}&timing=electronic&windReading=regular",
        "400mh": f"{root}/hurdles/400-metres-hurdles/outdoor/men/senior?{params}&timing=electronic",
        "3000mSC": f"{root}/middlelong/3000-metres-steeplechase/outdoor/men/senior?{params}",
        "high_jump": f"{root}/jumps/high-jump/outdoor/men/senior?{params}",
        "pole_vault": f"{root}/jumps/pole-vault/outdoor/men/senior?{params}",
        "long_jump": f"{root}/jumps/long-jump/outdoor/men/senior?{params}&windReading=regular",
        "triple_jump": f"{root}/jumps/triple-jump/outdoor/men/senior?{params}&windReading=regular",
        "shot_put": f"{root}/throws/shot-put/outdoor/men/senior?{params}",
        "discus": f"{root}/throws/discus-throw/outdoor/men/senior?{params}",
        "hammer": f"{root}/throws/hammer-throw/outdoor/men/senior?{params}",
        "javelin": f"{root}/throws/javelin-throw/outdoor/men/senior?{params}",
        "decathlon": f"{root}/combined-events/decathlon/outdoor/men/senior?{params}&windReading=regular"
    }[event]

def download_world_athletics(event):
    year_to_values = {}
    for year in range(1960, 2025):
        print(year)
        year_to_values[year] = download_world_athletics_year(year, event)
    return year_to_values

def download_world_athletics_year(year, event):
    times = []
    for page in range(1,6):
        r = requests.get(get_world_url(event,year,page))
        soup = BeautifulSoup(r.text,"lxml")
        rows = soup.select("table.records-table tr")
        row_times = [row.select("td")[1].text.strip() for row in rows[1:]]
        row_times = [EVENT_TO_SIGN[event] * parse_time(x) for x in row_times if re.match(r'^-?\d+(?:\.\d+)?$', x.replace(":",""))]
        if len(row_times) == 0: break
        times += row_times
    return times

if __name__ == "__main__":
    for event in EVENT_TO_SIGN:
        print(event)
        year_to_values = download_world_athletics(event)
        open(f"/tmp/records_{event}.json","w").write(json.dumps(year_to_values))
