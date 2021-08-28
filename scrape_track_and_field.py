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
        "800m": f"{root}/middle-long/800-metres/outdoor/men/senior?{params}&timing=electronic",
        "1500m": f"{root}/middle-long/1500-metres/outdoor/men/senior?{params}",
        "5000m": f"{root}/middle-long/5000-metres/outdoor/men/senior?{params}",
        "10000m": f"{root}/middle-long/10000-metres/outdoor/men/senior?{params}",
        "marathon": f"{root}/road-running/marathon/outdoor/men/senior?{params}&drop=regular&fiftyPercentRule=regular",
        "110mh": f"{root}/hurdles/110-metres-hurdles/outdoor/men/senior?{params}&timing=electronic&windReading=regular",
        "400mh": f"{root}/hurdles/400-metres-hurdles/outdoor/men/senior?{params}&timing=electronic",
        "3000mSC": f"{root}/middle-long/3000-metres-steeplechase/outdoor/men/senior?{params}",
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
    for year in range(1960, 2022):
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

def scrape_older_years():
    #pull these from
    pass

def get_alltime_url(event):
    return {
        "100m": f"http://www.alltime-athletics.com/m_100ok.htm",
        "200m": f"http://www.alltime-athletics.com/m_200ok.htm",
        "400m": f"http://www.alltime-athletics.com/m_400ok.htm",
        "800m": f"http://www.alltime-athletics.com/m_800ok.htm",
        "1500m": f"http://www.alltime-athletics.com/m_1500ok.htm",
        "5000m": f"http://www.alltime-athletics.com/m_5000ok.htm",
        "10000m": f"http://www.alltime-athletics.com/m_10kok.htm",
        "marathon": f"http://www.alltime-athletics.com/mmaraok.htm",
        "110mh": f"http://www.alltime-athletics.com/m_110hok.htm",
        "400mh": f"http://www.alltime-athletics.com/m_400hok.htm",
        "3000mSC": f"http://www.alltime-athletics.com/m3000hok.htm",
        "high_jump": f"http://www.alltime-athletics.com/mhighok.htm",
        "pole_vault": f"http://www.alltime-athletics.com/mpoleok.htm",
        "long_jump": f"http://www.alltime-athletics.com/mlongok.htm",
        "triple_jump": f"http://www.alltime-athletics.com/mtripok.htm",
        "shot_put": f"http://www.alltime-athletics.com/mshotok.htm",
        "discus": f"http://www.alltime-athletics.com/mdiscok.htm",
        "hammer": f"http://www.alltime-athletics.com/mhammok.htm",
        "javelin": f"http://www.alltime-athletics.com/mjaveok.htm",
        "decathlon": f"http://www.alltime-athletics.com/mdecaok.htm"
    }[event]

def download_alltime_athletics(event):
    url = get_alltime_url(event)
    r = requests.get(url)
    soup = BeautifulSoup(r.text,"lxml")
    lines = soup.select("pre")[0].text.split("\n")
    lines = [l.strip().split("  ") for l in lines if l.strip()]
    lines = [[x.strip() for x in l if x] for l in lines]
    if event in ["long_jump","triple_jump"]:
        #some rows are missing the 7th field -- add a placeholder
        lines = [l[:6] + [""] + l[6:] if len(l) == 8 else l for l in lines]
    lines = [l for l in lines if not "a" in l[1]] #Boston Marathon times labelled with "a" and think aren't usually counted
    scores = [{"score": EVENT_TO_SIGN[event] * parse_time(l[1]), "name": l[-6], "year": int(l[-1].split(".")[-1])} for l in lines]
    #one score per person per year:
    deduped_scores = []
    year_to_scores = {}
    name_year = set()
    for s in scores:
        key = (s["name"],s["year"])
        if s["year"] == 2015: #"Henderson" in s["name"]:
            print(key, s)
        if key in name_year:
            continue
        name_year.add(key)
        year_to_scores.setdefault(s["year"],[]).append(s["score"])
    return year_to_scores

def download_event(event):
    year_to_values = download_world_athletics(event)
    # year_to_values2 = download_alltime_athletics(event)

    # all_years = list(set(list(year_to_values1.keys()) + list(year_to_values2.keys())))
    # year_to_values = {}
    # for year in all_years:
    #     if year in year_to_values1 and year in year_to_values2:
    #         num_to_compare = min(5,len(year_to_values1[year]), len(year_to_values2[year]))
    #         if year_to_values1[year][:num_to_compare] != year_to_values2[year][:num_to_compare]:
    #             if (event,year) in [("400m",2009),("800m",2021),("1500m",2017),("110mh",2020),("3000mSC",2003),("high_jump",2001),("pole_vault",2004),("pole_vault",2009),("pole_vault",2010),("pole_vault",2011),("pole_vault",2012),("pole_vault",2021),("triple_jump",2011),("discus",2009),("hammer",2012),("hammer",2013),("hammer",2016),("hammer",2019),("javelin",2020),("decathlon",2015),("decathlon",2020)]:
    #                 continue #manually verified
    #             print("ERROR: mismatch between sources - ", event, year, year_to_values1[year], year_to_values2[year])
    #             raise
    #     if len(year_to_values1.get(year,[])) > len(year_to_values2.get(year,[])):
    #         year_to_values[year] = year_to_values1.get(year,[])
    #     else:
    #         year_to_values[year] = year_to_values2.get(year,[])
    open(f"/tmp/records_{event}.json","w").write(json.dumps(year_to_values))




if __name__ == "__main__":
    # year_to_values = {}
    # for event in EVENT_TO_SIGN:
    #     print(event)
    #     for year in range(2001, 2022): #including current year although incomplete
    #         year_to_values[year] = download_year(year, event)

    #     open(f"/tmp/records_{event}.json","w").write(json.dumps(year_to_values))
    for event in EVENT_TO_SIGN:
        print(event)
        #if event != "long_jump": continue
        #if event in ["100m","200m","400m","800m","1500m","5000m","10000m","marathon","110mh","400mh","3000mSC","high_jump","pole_vault","long_jump","triple_jump","shot_put","discus","hammer"]: continue #,"javelin"]: continue
        year_to_values = download_world_athletics(event)
        open(f"/tmp/records_{event}.json","w").write(json.dumps(year_to_values))
