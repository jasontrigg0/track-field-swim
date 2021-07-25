import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import scipy.optimize as opt
import math
import json
import re
import os

#trying to model a couple things:
#how impressive are various track and field world records?
#and how fast is the rate of improvement in the various disciplines?
#data is the top 500 athletes (ie one score per athlete) for each year from 2001-2021
#model is that the distribution each year is normal
#that all years have the same sd, and that the mean changes over time
#as A * Be^-kt + C
#results:
#the model doesn't look like the tail of a distribution with 6B people in it
#instead more like 1-10 thousand so maybe it's looking at an effective population
#of elite athletes? or maybe something is wrong...
#model seems to over-predict the likelihood of world records in a given year
#only reason I can think of is that the tails of the distribution are thinner
#than normal? strange issue and not sure how to address it
#in general the field events seem to be improving faster than the track events
#haven't looked into how the effective population size is growing over time
#maybe model that separately as some exponential growth?
#big TODO here is to resolve the issue where the normal curve is overpredicting the tail
#might want to switch models entirely

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
    "440mh": -1,
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

def event_to_url(event, year, page):
    return {
        "100m": f"https://www.worldathletics.org/records/toplists/sprints/100-metres/outdoor/men/senior/{year}?regionType=world&timing=electronic&windReading=regular&page={page}&bestResultsOnly=true",
        "200m": f"https://www.worldathletics.org/records/toplists/sprints/200-metres/outdoor/men/senior/{year}?regionType=world&timing=electronic&windReading=regular&page={page}&bestResultsOnly=true",
        "400m": f"https://www.worldathletics.org/records/toplists/sprints/400-metres/outdoor/men/senior/{year}?regionType=world&timing=electronic&page={page}&bestResultsOnly=true",
        "800m": f"https://www.worldathletics.org/records/toplists/middle-long/800-metres/outdoor/men/senior/{year}?regionType=world&timing=electronic&page={page}&bestResultsOnly=true",
        "1500m": f"https://www.worldathletics.org/records/toplists/middle-long/1500-metres/outdoor/men/senior/{year}?regionType=world&page={page}&bestResultsOnly=true",
        "5000m": f"https://www.worldathletics.org/records/toplists/middle-long/5000-metres/outdoor/men/senior/{year}?regionType=world&page={page}&bestResultsOnly=true",
        "10000m": f"https://www.worldathletics.org/records/toplists/middle-long/10000-metres/outdoor/men/senior/{year}?regionType=world&page={page}&bestResultsOnly=true",
        "marathon": f"https://www.worldathletics.org/records/toplists/road-running/marathon/outdoor/men/senior/{year}?regionType=world&drop=regular&fiftyPercentRule=regular&page={page}&bestResultsOnly=true",
        "110mh": f"https://www.worldathletics.org/records/toplists/hurdles/110-metres-hurdles/outdoor/men/senior/{year}?regionType=world&timing=electronic&windReading=regular&page={page}&bestResultsOnly=true",
        "440mh": f"https://www.worldathletics.org/records/toplists/hurdles/400-metres-hurdles/outdoor/men/senior/{year}?regionType=world&timing=electronic&page={page}&bestResultsOnly=true",
        "3000mSC": f"https://www.worldathletics.org/records/toplists/middle-long/3000-metres-steeplechase/outdoor/men/senior/{year}?regionType=world&page={page}&bestResultsOnly=true",
        "high_jump": f"https://www.worldathletics.org/records/toplists/jumps/high-jump/outdoor/men/senior/{year}?regionType=world&page={page}&bestResultsOnly=true",
        "pole_vault": f"https://www.worldathletics.org/records/toplists/jumps/pole-vault/outdoor/men/senior/{year}?regionType=world&page={page}&bestResultsOnly=true",
        "long_jump": f"https://www.worldathletics.org/records/toplists/jumps/long-jump/outdoor/men/senior/{year}?regionType=world&windReading=regular&page={page}&bestResultsOnly=true",
        "triple_jump": f"https://www.worldathletics.org/records/toplists/jumps/triple-jump/outdoor/men/senior/{year}?regionType=world&windReading=regular&page={page}&bestResultsOnly=true",
        "shot_put": f"https://www.worldathletics.org/records/toplists/throws/shot-put/outdoor/men/senior/{year}?regionType=world&page={page}&bestResultsOnly=true",
        "discus": f"https://www.worldathletics.org/records/toplists/throws/discus-throw/outdoor/men/senior/{year}?regionType=world&page={page}&bestResultsOnly=true",
        "hammer": f"https://www.worldathletics.org/records/toplists/throws/hammer-throw/outdoor/men/senior/{year}?regionType=world&page={page}&bestResultsOnly=true",
        "javelin": f"https://www.worldathletics.org/records/toplists/throws/javelin-throw/outdoor/men/senior/{year}?regionType=world&page={page}&bestResultsOnly=true",
        "decathlon": f"https://www.worldathletics.org/records/toplists/combined-events/decathlon/outdoor/men/senior/{year}?regionType=world&windReading=regular&page={page}&bestResultsOnly=true"
    }[event]

def parse_time(time):
    #https://stackoverflow.com/a/41252517
    return sum(x * float(t) for x, t in zip([1, 60, 3600], time.split(":")[::-1]))

def fit_performance_trend(year_to_values, sign=1):
    #NOTE: assuming we have the top k performances by year (one per athlete) but allowing k to vary by year
    #model is that each year performances are distributed normally, and that there are three factors that may vary by year:
    #- mean
    #- sd
    #- number of participants / number of samples from the distribution
    # model assumes that the mean is decreasing towards a horizontal asymptote
    # the sd is constant
    # no assumptions about the number of participants (add this in later? maybe use to predict wr likelihood going forward?)
    # mean = x0 * e^x1*t + x2
    # sd = x3

    def cost_function(theta):
        print(theta)
        total_cost = 0
        for year in year_to_values:
            values = year_to_values[year]
            loc = theta[0] * math.e ** (-1 * theta[1] * (year-1900)) + theta[2] #mean increasing by year
            scale = theta[3] #sd constant and specified
            xmin = min(values)
            xmax = max(values)
            a = (xmin - loc) / scale
            b = 1000 #no boundary to the right #(xmax - loc) / scale
            print(year, loc, scale, a, b)
            #print(year, loc, scale)
            year_cost = -scipy.stats.truncnorm(a, b, loc=loc, scale=scale).logpdf(values)
            total_cost += year_cost.sum() / len(values) #equalize cost for each year, does this make sense?
        return total_cost

    #calculate a first guess at the parameters
    wr = max([x for year in year_to_values for x in year_to_values[year]])
    sd = np.std([x for year in year_to_values for x in year_to_values[year]])

    #(wild) initial guess:
    #we're 4 sds below the asymptote at t0
    #that gap is shrinking about 1% per year
    #asymptote of the elite athlete mean is the current world record
    #sd of the elite athlete field is the same as the sd of all performances we've seen
    return opt.fmin(cost_function, [-4*sd, 0.01, wr, sd], maxfun = 100)



def fit_truncnorm(values, fa=None, fb=None):
    #the builtin truncnorm.fit function seems kinda terrible / doesn't often converge to reasonable values
    #return scipy.stats.truncnorm.fit(values, scale=scale_guess, loc=loc_guess)
    #instead do a quick convergence

    #fa, fb allow freezing the parameters a and b
    #(but the distribution is still forced to contain all the datapoints)

    xmin, xmax = min(values), max(values)

    def cost_function(theta):
        #specifies loc and scale
        #we use those plus the range of the data
        #x_min, x_max
        #to derive the a and b and score
        loc, scale = theta
        a = (xmin - loc) / scale
        b = (xmax - loc) / scale
        if fa is not None:
            a = min(fa,a)
        if fb is not None:
            b = max(fb,b)
        cost = -scipy.stats.truncnorm(a, b, loc=loc, scale=scale).logpdf(values)
        return cost.sum()

    loc, scale = opt.fmin(cost_function, [0,1])
    a = (xmin - loc) / scale
    b = (xmax - loc) / scale
    if fa is not None:
        a = min(fa,a)
    if fb is not None:
        b = max(fb,b)
    return [a, b, loc, scale]

def download_year(year, event):
    times = []
    for page in range(1,6):
        r = requests.get(event_to_url(event,year,page))
        soup = BeautifulSoup(r.text,"lxml")
        rows = soup.select("table.records-table tr")
        row_times = [row.select("td")[1].text.strip() for row in rows[1:]]
        row_times = [EVENT_TO_SIGN[event] * parse_time(x) for x in row_times if re.match(r'^-?\d+(?:\.\d+)?$', x.replace(":",""))]
        times += row_times
    return times

def plot_year(year, event, times = None):
    if times is None:
        times = download_year(year, event)
    fig, ax = plt.subplots(1, 1)
    ax.hist(times, density=True, histtype='stepfilled', alpha=0.2)

    #add truncnorm fit
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    par = fit_truncnorm(times, fb=1000)
    loc, scale = par[2], par[3]
    print(loc, scale)
    ax.plot(x, scipy.stats.truncnorm.pdf(x, *par), 'b-', lw=2)
    #ax.legend(loc='best', frameon=False)
    plt.show()

def process_event(event, plot=False):
    if os.path.isfile(f"/tmp/records_{event}.json"):
        year_to_values = json.load(open(f'/tmp/records_{event}.json'))
        year_to_values = {int(k):v for k,v in year_to_values.items()}
    else:
        year_to_values = {}
        for year in range(2001, 2022): #including current year although incomplete
            print(year)
            year_to_values[year] = download_year(year, event)
        open(f"/tmp/records_{event}.json","w").write(json.dumps(year_to_values))

    if os.path.isfile(f"/tmp/params.json"):
        all_params = json.load(open(f'/tmp/params.json'))
    else:
        all_params = {}
    if event in all_params:
        params = all_params[event]
    else:
        params = fit_performance_trend(year_to_values)
        all_params[event] = list(params)
        open(f"/tmp/params.json","w").write(json.dumps(all_params))


    #compute the number of people in 2019 (exclude 2020 because of covid and 2021 because it's incomplete)
    cnt_year = 2019
    loc = params[0] * math.e ** (-1 * params[1] * (cnt_year - 1900)) + params[2]
    scale = params[3]
    cnt = len(year_to_values[cnt_year]) / scipy.stats.norm.sf(min(year_to_values[cnt_year]), loc, scale)

    #compute wr probability for this year
    current_year = 2021 #2046
    loc = params[0] * math.e ** (-1 * params[1] * (current_year - 1900)) + params[2] #TODO: create a class for the model
    scale = params[3]

    #hard code wrs from before the dataset started in 2001
    event_wrs = {
        "1500m": -206,
        "high_jump": 2.45,
        "pole_vault": 6.18, #we're looking at outdoor but record was indoor
        "long_jump": 8.95,
        "triple_jump": 18.29,
        "discus": 74.08,
        "hammer": 86.74,
        "javelin": 98.48
    }
    wr = max([x for year in year_to_values for x in year_to_values[year]])
    if event in event_wrs:
        wr = max(event_wrs[event], wr)

    rate = -1 * params[0] * params[1] * math.e ** (-1 * params[1] * (current_year - 1900)) / abs(wr)
    exp_wr = cnt * scipy.stats.norm.sf(wr, loc, scale) #expected # of world record times in 2021
    print(",".join([event, str(rate), str(exp_wr)]))

    #alternatively: fit on 2019 only
    if plot:
        year_to_graph = 2021
        par = fit_truncnorm(year_to_values[year_to_graph], fb=1000) #test out 2019 directly
        loc, scale = par[2], par[3]
        exp_wr = len(year_to_values[cnt_year]) * scipy.stats.norm.sf(wr, loc, scale) / scipy.stats.norm.sf(min(year_to_values[year_to_graph]), loc, scale)
        print(event, exp_wr) #prob of world record in plot_year from plot_year only
        plot_year(year_to_graph, event, year_to_values[year_to_graph])



if __name__ == "__main__":
    # for event in ['3000mSC', 'hammer', 'javelin']:
    #     process_event(event, True)
    for event in EVENT_TO_SIGN:
        process_event(event)
