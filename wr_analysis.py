import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import scipy.optimize as opt
import math
import json
import os
import sklearn.linear_model
import sklearn.metrics
import pandas as pd
import sys

#TODO: remove
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

SWIMMING_EVENTS = [
    "50freestyle",
    "100freestyle",
    "200freestyle",
    "400freestyle",
    "800freestyle",
    "1500freestyle",
    "100backstroke",
    "200backstroke",
    "100breaststroke",
    "200breaststroke",
    "100butterfly",
    "200butterfly",
    "200medley",
    "400medley"
]

FIELD_EVENTS = ["high_jump","pole_vault","long_jump","triple_jump","shot_put","discus","hammer","javelin","decathlon"]

def get_all_events():
    # return SWIMMING_EVENTS
    return list(EVENT_TO_SIGN.keys()) + SWIMMING_EVENTS

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


#TODO: each year fit the estimated peak as predicted by that year
#then take these points and fit an exponential decay against them
#lastly examine how the actual peaks compare to the estimated peaks,
#are they always off by a fixed amount?
#can compare against a direct regression of the fastest time each year
#to see how noisy that is

def fit_peak_trend2(year_to_scores):
    print("year,cnt,mean,sd,n,min,x256,x128,x64,x32,x16,x8,x4,x2,best,exp")
    for year in year_to_scores:
        scores = year_to_scores[year]
        par = fit_truncnorm(scores, fb=1000)
        _, _, loc, scale = par
        #print(par)
        cnt = len(scores)
        xmin = min(scores)
        x256 = scores[256] if len(scores) > 256 else ""
        x128 = scores[128] if len(scores) > 128 else ""
        x64 = scores[64] if len(scores) > 64 else ""
        x32 = scores[32] if len(scores) > 32 else ""
        x16 = scores[16] if len(scores) > 16 else ""
        x8 = scores[8] if len(scores) > 8 else ""
        x4 = scores[4]
        x2 = scores[2]
        #we know there were len(scores) values above xmin
        #and want to choose a value for peak such that
        #we expect 1 value above peak
        peak_frac = scipy.stats.norm.sf(xmin, loc, scale) / cnt
        peak = scipy.stats.norm.isf(peak_frac, loc, scale)
        print(",".join([str(year),str(cnt),str(loc),str(scale),str(1/peak_frac),str(xmin),str(x256),str(x128),str(x64),str(x32),str(x16),str(x8),str(x4),str(x2),str(max(scores)),str(peak)]))


def fit_peak_trend(year_to_scores):
    #NOTE: assuming we have the top k performances by year (one per athlete) but allowing k to vary by year
    # peak performance = x0 * e^x1*t + x2

    def cost_function(theta):
        print(theta)
        total_cost = 0
        for year in year_to_scores:
            scores = year_to_scores[year]
            #re-parametrizing:
            #peak = theta[0] * math.e ** (-1 * theta[1] * (year-2000)) + theta[2] #peak increasing by year

            #peak = x0 + x1 * e**(-1*x2*(year-2000))
            #-> if we set theta[0], theta[1], theta[2]
            #to the asymp, deriv, second deriv
            #value_2000 = x0 + x1
            #deriv_2000 = -1 * x1 * x2
            #second_deriv_2000 = x1 * x2**2
            x1 = theta[1]**2 / theta[2]
            x2 = -1 * theta[2] / theta[1]
            x0 = theta[0] - x1
            peak = x0 + x1 * math.e ** (-1 * x2 * (year - 2010))
            xmin = min(scores)
            xmax = max(scores)
            cnt = len(scores)
            def normal_cost_fn(theta):
                #compute normal distribution with
                #len(scores) values greater than or equal to xmin
                #and 1 value greater than or equal to peak
                loc, scale = theta
                ratio = scipy.stats.norm.sf(xmin, loc, scale) / scipy.stats.norm.sf(peak, loc, scale)
                return (math.log(ratio) - math.log(cnt))**2
            loc_guess = 2*xmin - peak
            scale_guess = (peak - xmin) / 2 #2sds from xmin to the peak
            loc, scale = opt.fmin(normal_cost_fn, [loc_guess, scale_guess], disp=False)
            print(year, loc, scale, peak)
            a = (xmin - loc) / scale
            b = 1000 #no boundary to the right #(xmax - loc) / scale
            year_cost = -scipy.stats.truncnorm(a, b, loc=loc, scale=scale).logpdf(scores)
            total_cost += year_cost.sum() / len(scores) #equalize cost for each year, does this make sense?
        print(total_cost)
        if math.isnan(total_cost):
            return 0
        return total_cost

    #calculate a first guess at the parameters
    wr = max([x for year in year_to_scores for x in year_to_scores[year]])
    sd = np.std([x for year in year_to_scores for x in year_to_scores[year]])

    #(wild) initial guess:
    #we're 4 sds below the asymptote at t0
    #gap is shrinking about 1% per year
    #asymptote is 3 sds above current wr
    # return opt.fmin(cost_function, [-10*sd, 0.1, wr + 5*sd], maxfun = 1000, ftol=0.00001)
    return opt.minimize(cost_function, [wr, 0.1*sd, -0.001*sd], method='BFGS')


def fit_performance_trend(year_to_scores):
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
        for year in year_to_scores:
            scores = year_to_scores[year]
            loc = theta[0] * math.e ** (-1 * theta[1] * (year-1900)) + theta[2] #mean increasing by year
            scale = theta[3] #sd constant and specified
            xmin = min(scores)
            xmax = max(scores)
            a = (xmin - loc) / scale
            b = 1000 #no boundary to the right #(xmax - loc) / scale
            print(year, loc, scale, a, b)
            #print(year, loc, scale)
            year_cost = -scipy.stats.truncnorm(a, b, loc=loc, scale=scale).logpdf(scores)
            total_cost += year_cost.sum() / len(scores) #equalize cost for each year, does this make sense?
        return total_cost

    #calculate a first guess at the parameters
    wr = max([x for year in year_to_scores for x in year_to_scores[year]])
    sd = np.std([x for year in year_to_scores for x in year_to_scores[year]])

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

    loc, scale = opt.fmin(cost_function, [0,1], disp=False)
    a = (xmin - loc) / scale
    b = (xmax - loc) / scale
    if fa is not None:
        a = min(fa,a)
    if fb is not None:
        b = max(fb,b)
    return [a, b, loc, scale]

def plot_year(year, event, scores = None):
    if scores is None:
        scores = download_year(year, event)
    fig, ax = plt.subplots(1, 1)
    ax.hist(scores, density=True, histtype='stepfilled', alpha=0.2)

    #add truncnorm fit
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    par = fit_truncnorm(scores, fb=1000)
    loc, scale = par[2], par[3]
    print(loc, scale)
    ax.plot(x, scipy.stats.truncnorm.pdf(x, *par), 'b-', lw=2)
    #ax.legend(loc='best', frameon=False)
    plt.show()

def process_event(event, plot=False):
    year_to_scores = json.load(open(f'/tmp/records_{event}.json'))
    year_to_scores = {int(k):v for k,v in year_to_scores.items()}

    # if os.path.isfile(f"/tmp/params.json"):
    #     all_params = json.load(open(f'/tmp/params.json'))
    # else:
    #     all_params = {}
    # if event in all_params:
    #     params = all_params[event]
    # else:
    #     params = fit_performance_trend(year_to_scores)
    #     all_params[event] = list(params)
    #     open(f"/tmp/params.json","w").write(json.dumps(all_params))

    params = fit_performance_trend(year_to_scores)

    #compute the number of people in 2019 (exclude 2020 because of covid and 2021 because it's incomplete)
    cnt_year = 2019
    loc = params[0] * math.e ** (-1 * params[1] * (cnt_year - 1900)) + params[2]
    scale = params[3]
    cnt = len(year_to_scores[cnt_year]) / scipy.stats.norm.sf(min(year_to_scores[cnt_year]), loc, scale)

    #compute wr probability for this year
    current_year = 2021
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
        "javelin": 98.48,
        "50freestyle": -20.91,
        "100freestyle": -46.91,
        "200freestyle": -102,
        "400freestyle": -220.07,
        "800freestyle": -452.12,
        "200backstroke": -111.92,
        "400medley": -243.84
    }
    wr = max([x for year in year_to_scores for x in year_to_scores[year]])
    if event in event_wrs:
        wr = max(event_wrs[event], wr)

    rate = -1 * params[0] * params[1] * math.e ** (-1 * params[1] * (current_year - 1900)) / abs(wr)
    exp_wr = cnt * scipy.stats.norm.sf(wr, loc, scale) #expected # of world record scores in 2021
    print(",".join([event, str(rate), str(exp_wr)]))

    #alternatively: fit on 2019 only
    if plot:
        year_to_graph = 2021
        par = fit_truncnorm(year_to_scores[year_to_graph], fb=1000) #test out 2019 directly
        loc, scale = par[2], par[3]
        exp_wr = len(year_to_scores[cnt_year]) * scipy.stats.norm.sf(wr, loc, scale) / scipy.stats.norm.sf(min(year_to_scores[year_to_graph]), loc, scale)
        for i, score in enumerate(year_to_scores[year_to_graph][::-1]):
            exp_score = len(year_to_scores[cnt_year]) * scipy.stats.norm.sf(score, loc, scale) / scipy.stats.norm.sf(min(year_to_scores[year_to_graph]), loc, scale)
            print(score, len(year_to_scores[year_to_graph]) - i, exp_score)
        print("expected number of wr scores:")
        print(event, exp_wr) #prob of world record in plot_year from plot_year only
        plot_year(year_to_graph, event, year_to_scores[year_to_graph])

MAX_YEAR = 2021

def exp_fit(years, vals):
    def pred_fn(theta, year):
        x0, x1, x2 = theta
        return x0 + x1 * math.e**(-1 * x2 * (year - MAX_YEAR))

    def cost_function(theta):
        x0, x1, x2 = theta
        for y, v in zip(years, vals):
            pred = pred_fn(theta, y)
            cost += (pred - x) ** 2
        return cost

    #compute guess that fits the first and last points
    guess_x1 = (vals[-1] - vals[0]) / (math.e ** (-1 * 0.03 * (years[-1] - MAX_YEAR)) - math.e ** (-1 * 0.03 * (years[0] - MAX_YEAR)))
    guess_x0 = vals[0] - guess_x1 * math.e ** (-1 * 0.03 * (years[0] - MAX_YEAR))

    guess = [guess_x0, guess_x1, 0.03]
    parameters = opt.minimize(cost_function, guess, method='BFGS')

def get_normalized_scores(event):
    year_to_scores = json.load(open(f'/tmp/records_{event}.json'))
    year_to_scores = {int(k):v for k,v in year_to_scores.items()}

    best_curr = max(year_to_scores[MAX_YEAR])
    year_to_scores = {int(k):[x/abs(best_curr) for x in v] for k,v in year_to_scores.items()}
    #after the above most scores are ~1 or ~-1
    #convert all to ~1 to avoid sign issues
    year_to_scores = {int(k):[2+x if x<0 else x for x in v] for k,v in year_to_scores.items()}

    if event in SWIMMING_EVENTS:
        #exclude the supersuit years: 2008+2009
        year_to_scores = {k:v for k,v in year_to_scores.items() if k not in [2008,2009]}

    return year_to_scores


def regress_yearly_best(event):
    year_to_scores = get_normalized_scores(event)

    df = pd.DataFrame({"year":[y for y in year_to_scores]})
    df["x1"] = df.apply(lambda x: year_to_scores[x["year"]][0],axis=1)
    data = []

    for y in year_to_scores:
        df_regress = df.copy()
        #df_regress = df_regress[df_regress["year"] != y]
        df_regress = df_regress[(df_regress["year"] < y-3) | (df_regress["year"] > y+3)]

        #linreg
        reg = sklearn.linear_model.LinearRegression()
        reg.fit(df_regress[["year"]], df_regress["x1"])
        # print(reg.coef_)
        # print(reg.intercept_)
        linreg_pred = reg.predict([[y]])[0]

        #lasso
        reg = sklearn.linear_model.LassoCV(normalize=True)
        reg.fit(df_regress[["year"]], df_regress["x1"])
        # print(reg.coef_)
        # print(reg.intercept_)
        lasso_pred = reg.predict([[y]])[0]

        #exponential
        exp = ExpModel(max_exp = 0.1)
        exp.fit(df_regress["year"].values, df_regress["x1"].values)
        # print(exp.params)
        exp_pred = exp.predict([y])[0]
        if (exp_pred-1)**2 > 1:
            print(df_regress["year"].values, df_regress["x1"].values)
            print(exp.params)
            raise

        expmix = ExpMixModel(max_exp = 0.1)
        expmix.fit(df_regress["year"].values, df_regress["x1"].values)
        # print(exp.params)
        expmix_pred = expmix.predict([y])[0]

        data.append({
            "event": event,
            "year": y,
            "x1": year_to_scores[y][0],
            "linreg": linreg_pred,
            "lasso": lasso_pred,
            "exp": exp_pred,
            "expmix": expmix_pred
        })

    return data

def project_yearly_best(event, curr_year=2014, horizon=5):
    #using data before curr_year, try to predict curr_year + 5
    #excluding 2020 because of covid, and leaving a 5 year gap
    #to try to exclude the likelihood that the same athlete is
    #best in both 2014 and 2019 (does this matter?)
    year_to_scores = json.load(open(f'/tmp/records_{event}.json'))
    year_to_scores = {int(k):v for k,v in year_to_scores.items()}

    #normalize data by the best score of 2014 so that all events are weighted equally
    fut_year = curr_year + horizon
    if len(year_to_scores.get(curr_year,[])) < 8:
        return None
    if len(year_to_scores.get(fut_year,[])) < 8:
        return None
    best_curr = max(year_to_scores[curr_year])
    x8_curr = year_to_scores[curr_year][7]
    best_fut = year_to_scores[fut_year][0]
    x8_fut = year_to_scores[fut_year][7]
    year_to_scores = {int(k):[x/abs(best_curr) for x in v] for k,v in year_to_scores.items()}
    #after the above most scores are ~1 or ~-1
    #convert all to ~1 to avoid sign issues below
    year_to_scores = {int(k):[2+x if x<0 else x for x in v] for k,v in year_to_scores.items()}
    gap_fut = (year_to_scores[fut_year][7] - year_to_scores[fut_year][0]) # + year_to_scores[2018][7] + year_to_scores[2017][7] - year_to_scores[2019][0] - year_to_scores[2018][0] - year_to_scores[2017][0]) / 3

    years = range(1980,curr_year+1)
    if event in SWIMMING_EVENTS:
        #exclude the supersuit years: 2008+2009
        years = [x for x in years if x not in [2008,2009]]

    #exclude years with minimal data
    years = [y for y in years if len(year_to_scores.get(y,[])) >= 8]
    if len(years) < 10: #require 10 years of data to get proper trends
        return None
    df = pd.DataFrame({"year":years})

    def get_curve_best(scores):
        print("here")
        par = fit_truncnorm(scores, fb=1000)
        _, _, loc, scale = par
        cnt = len(scores)
        xmin = min(scores)
        #we know there were len(scores) values above xmin
        #and want to choose a value for peak such that
        #we expect 1 value above peak
        peak_frac = scipy.stats.norm.sf(xmin, loc, scale) / cnt
        peak = scipy.stats.norm.isf(peak_frac, loc, scale)
        return peak


    # df["curve_best"] = df.apply(lambda x: get_curve_best(year_to_scores[x["year"]]),axis=1)
    df["x1"] = df.apply(lambda x: year_to_scores[x["year"]][0],axis=1)
    df["x2"] = df.apply(lambda x: year_to_scores[x["year"]][1],axis=1)
    df["x4"] = df.apply(lambda x: year_to_scores[x["year"]][3],axis=1)
    df["x8"] = df.apply(lambda x: year_to_scores[x["year"]][7],axis=1)
    df["x16"] = df.apply(lambda x: year_to_scores[x["year"]][15] if len(year_to_scores[x["year"]]) >=16 else min(year_to_scores[x["year"]]),axis=1)
    df["x32"] = df.apply(lambda x: year_to_scores[x["year"]][31] if len(year_to_scores[x["year"]]) >=32 else min(year_to_scores[x["year"]]),axis=1)
    df["gap"] = (df["x8"] / df["x1"] - 1)

    reg = sklearn.linear_model.LinearRegression() #LassoCV(normalize=True) #LinearRegression() #LassoCV(normalize=True) #sklearn.linear_model.LinearRegression()
    reg.fit(df[["year"]], df["x1"], df.apply(lambda x: math.e**((x["year"]-2020)/10), axis=1))

    df["line_x1"] = reg.predict(df[["year"]]) - 1
    df["est_gap"] = (df["x8"] / (df["line_x1"] + 1) - 1)

    data = {
        "event": event,
        "best_fut": ((best_fut - best_curr) / abs(best_curr)),
        "x8_fut": ((x8_fut - best_curr) / abs(best_curr)),
        "gap_fut": gap_fut,
        "raw_fut": best_fut,
        "raw_curr": best_curr,
        "raw_x8": x8_curr,
        "fit_curr": reg.predict([[curr_year]])[0] - 1,
        "gap_avg": df["gap"].mean(),
        "gap_est_avg": df["est_gap"].mean(),
        "gap_curr": df[df["year"] == curr_year]["gap"].values[0],
        "gap_est_curr": df[df["year"] == curr_year]["est_gap"].values[0],
        "year": curr_year,
        "exp_year": math.e ** (-0.003 * curr_year),
        "exp_year_2": math.e ** (-0.01 * curr_year)
    }

    #print(data)

    # reg = sklearn.linear_model.LinearRegression() #LassoCV(normalize=True) #LinearRegression()
    # reg.fit(df[["year"]], df[f"curve_best"])
    # data[f"trend_curve"] = reg.coef_[0] * 5 #expected change from 2014 - 2019

    for n in [1,2,4,8,16,32]:
        reg = sklearn.linear_model.LinearRegression() #LassoCV(normalize=True) #LinearRegression()
        reg.fit(df[["year"]], df[f"x{n}"])
        data[f"trend_{n}"] = reg.coef_[0] * 5 #expected change from 2014 - 2019
        yr_range = max(df["year"]) - min(df["year"])
        data[f"trend_{n}_b"] = reg.coef_[0] * 5 * yr_range
        data["yr_range"] = yr_range
        reg.fit(df[["year"]], df[f"x{n}"], df.apply(lambda x: math.e**((x["year"]-2020)/10), axis=1))
        data[f"trend_{n}_a"] = reg.coef_[0] * 5 #expected change from 2014 - 2019
        if n == 8:
            df["est2_gap"] = (reg.predict(df[["year"]]) / (df["line_x1"] + 1) - 1)

    data["gap_est2_avg"] = df["est2_gap"].mean()
    data["gap_est2_curr"] = df[df["year"] == curr_year]["est2_gap"].values[0]

    return data

def performance_outliers():
    curr_year = 2021
    events = get_all_events()
    data = []
    for e in events:
        year_to_scores = json.load(open(f"/tmp/records_{e}.json"))
        year_to_scores = {int(k):v for k,v in year_to_scores.items()}
        years = range(1980,curr_year+1)
        if e in SWIMMING_EVENTS:
            #exclude the supersuit years: 2008+2009
            years = [x for x in years if x not in [2008,2009]]
        years = [y for y in years if len(year_to_scores.get(y,[])) >= 8]
        print(years)
        reg = sklearn.linear_model.LassoCV(normalize=True)
        reg.fit([[y] for y in years], [year_to_scores[y][0] for y in years])
        preds = reg.predict([[y] for y in years])
        deltas = [val - pred for (pred,val) in zip(preds,[year_to_scores[y][0] for y in years])]
        sd_err = np.std(deltas)
        print(preds)
        print(deltas)
        print(sd_err)
        data += [{"year": y, "event": e, "val": d/sd_err} for y,d in zip(years,deltas)]
    data.sort(key=lambda x: x["val"], reverse=True)
    for x in data[:50]:
        print(x)
    raise

def fit_record_improvement(curr_year=2019, events = None):
    df = pd.DataFrame()

    if events is None:
        events = get_all_events()

    for e in events:
        print(e)
        year_to_scores = json.load(open(f'/tmp/records_{e}.json'))
        year_to_scores = {int(k):v for k,v in year_to_scores.items()}

        best_curr = max(year_to_scores[curr_year])
        year_to_scores = {int(k):[x/abs(best_curr) for x in v] for k,v in year_to_scores.items()}
        #after the above most scores are ~1 or ~-1
        #convert all to ~1 to avoid sign issues
        year_to_scores = {int(k):[2+x if x<0 else x for x in v] for k,v in year_to_scores.items()}

        years = range(1980,curr_year+1)
        if e in SWIMMING_EVENTS:
            #exclude the supersuit years: 2008+2009
            years = [x for x in years if x not in [2008,2009]]

        #exclude years with minimal data
        years = [y for y in years if len(year_to_scores.get(y,[])) >= 8]
        print(years)

        reg = sklearn.linear_model.LinearRegression()
        reg.fit([[y] for y in years], [year_to_scores[y][0] for y in years])
        trend_x1 = reg.coef_[0]
        if curr_year < 2010: raise #need to edit fit_mid below before using older years
        fit_x1_mid = reg.predict([[2010]])[0]
        fit_x1_curr = reg.predict([[curr_year]])[0]

        reg.fit([[y] for y in years], [year_to_scores[y][7] for y in years])
        trend_x8 = reg.coef_[0]
        fit_x8_mid = reg.predict([[2010]])[0]

        for y in years:
            data = {
                "event": e,
                "year": y,
                "gap": (fit_x8_mid / fit_x1_mid - 1),
                "trend_x1": trend_x1,
                "trend_x8": trend_x8,
                "fit_curr": fit_x1_curr,
                "swim": 1 * (e in SWIMMING_EVENTS),
                "field": 1 * (e in FIELD_EVENTS),
                "best": year_to_scores[y][0]
            }
            df = df.append(data, ignore_index=True)

    def linear_pred_fn(theta, r, year):
        x0, x1 = theta
        return x0 + x1*(year - curr_year)

    def exp_pred_fn(theta, r, year):
        x0, x1, x2, x3, x4, x5, x6 = theta
        return x0 + x1*r["swim"] + x2*r["field"] + (x3 + x4 * r["swim"] + x5 * r["field"]) * math.e**(-1 * x6 * (year - curr_year))

    pred_fn = exp_pred_fn

    def cost_function(theta):
        #improvement rate for an event
        #seems to be proportional to
        #x1 + x2*(gap between 1st and 8th)
        #so try an exponential fit that will
        #include this information
        #x0 + (x1+x2*gap) * e**(-1*x3*year-2000)
        time_decay_const = -0.003
        cost = 0
        for _,r in df.iterrows():
            time_decay = math.e ** (time_decay_const * (curr_year - r["year"]))
            #pred = x0 + x1*r["fit_curr"] + x2*r["trend"] + (x3 + x4 * r["trend"]) * math.e**(-1 * x5 * (r["year"] - curr_year))
            pred = pred_fn(theta, r, r["year"])
            cost += time_decay * (pred - r["best"]) ** 2
        print(cost)
        return cost
    guess = [0.1, 1.0, -700, -0.07, 700, 0.03]
    guess = [ 1, 0.002 ]
    guess = [ 1.12433668,  0.19049096, -0.14999564, -0.11950254, -0.19078407,  0.15256575,  0.00513088]
    parameters = opt.minimize(cost_function, guess, method='BFGS')

    print(parameters.x)
    df["pred"] = df.apply(lambda x: pred_fn(parameters.x, x, x["year"]), axis=1)

    preds = {}
    for e in events:
        data = df[(df["event"] == e) & (df["year"] == curr_year)].to_dict('records')[0]
        preds[e] = pred_fn(parameters.x, data, 2019)
    return preds

class ExpModel():
    def __init__(self, max_exp = None):
        self.max_exp = max_exp
    def exp(self, x, a, b, c):
        return a+b*np.exp(-c*(x-MAX_YEAR))
    def fit(self, x, y):
        guess_c = 0.003
        guess_b = (y[-1] - y[0]) / (math.e ** (-1 * guess_c * (x[-1] - MAX_YEAR)) - math.e ** (-1 * guess_c * (x[0] - MAX_YEAR)))
        guess_a = y[0] - guess_b * math.e ** (-1 * guess_c * (x[0] - MAX_YEAR))
        if self.max_exp is not None:
            bounds = ([-np.inf,-np.inf,-self.max_exp],[np.inf,np.inf,self.max_exp])
        else:
            bounds = ([-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf])
        self.params, _ = opt.curve_fit(self.exp, x, y, p0=(guess_a,guess_b,guess_c), bounds=bounds, maxfev=50000)
    def predict(self, x):
        return [self.exp(a, *self.params) for a in x]

class ExpMixModel():
    def __init__(self, max_exp = None):
        self.max_exp = max_exp
    def exp(self, x, a, b, c, d, e):
        return a + b*np.exp(-c*(x-MAX_YEAR)) + d*np.exp(-e*(x-MAX_YEAR))
    def fit(self, x, y):
        guess_e = 0.003
        guess_d = (y[-1] - y[0]) / (math.e ** (-1 * guess_e * (x[-1] - MAX_YEAR)) - math.e ** (-1 * guess_e * (x[0] - MAX_YEAR)))
        guess_c = guess_e
        guess_b = guess_d
        guess_a = y[0] - guess_b * math.e ** (-1 * guess_c * (x[0] - MAX_YEAR))
        guess = (guess_a,guess_b,guess_c,guess_d,guess_e)
        if self.max_exp is not None:
            bounds = ([-np.inf,-np.inf,-self.max_exp,-np.inf,-self.max_exp],[np.inf,np.inf,self.max_exp,np.inf,self.max_exp])
        else:
            bounds = ([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf])
        self.params, _ = opt.curve_fit(self.exp, x, y, p0=guess, bounds=bounds, maxfev=50000)
    def predict(self, x):
        return [self.exp(a, *self.params) for a in x]

def find_best_fit():
    #test linear, lasso linear, exponential, and mixture of two exponentials
    #to see which best fits the data
    #tentatively looks like exponential fits best (while mixture overfits), though
    #mixture broke on the swimming events
    df = pd.DataFrame()
    for event in get_all_events():
        print(event)
        df = df.append(regress_yearly_best(event), ignore_index=True)
    print(df)
    df.to_csv("/tmp/reg.csv", index=False)
    reg = sklearn.linear_model.LassoCV(normalize=True)
    reg.fit(df[["linreg"]], df["x1"])
    pred = reg.predict(df[["linreg"]])
    print(reg.coef_)
    print(reg.intercept_)
    print(sklearn.metrics.mean_squared_error(df["x1"],pred))
    print(sklearn.metrics.r2_score(pred,df["x1"]))
    print(sklearn.metrics.r2_score(df["linreg"],df["x1"]))
    reg.fit(df[["lasso"]], df["x1"])
    pred = reg.predict(df[["lasso"]])
    print(reg.coef_)
    print(reg.intercept_)
    print(sklearn.metrics.mean_squared_error(df["x1"],pred))
    print(sklearn.metrics.r2_score(pred,df["x1"]))
    print(sklearn.metrics.r2_score(df["lasso"],df["x1"]))
    reg.fit(df[["exp"]], df["x1"])
    pred = reg.predict(df[["exp"]])
    print(reg.coef_)
    print(reg.intercept_)
    print(sklearn.metrics.mean_squared_error(df["x1"],pred))
    print(sklearn.metrics.r2_score(pred,df["x1"]))
    print(sklearn.metrics.r2_score(df["exp"],df["x1"]))
    reg.fit(df[["expmix"]], df["x1"])
    #skipping expmix, it's slow and doesn't always converge
    # pred = reg.predict(df[["expmix"]])
    # print(reg.coef_)
    # print(reg.intercept_)
    # print(sklearn.metrics.mean_squared_error(df["x1"],pred))
    # print(sklearn.metrics.r2_score(pred,df["x1"]))
    # print(sklearn.metrics.r2_score(df["expmix"],df["x1"]))
    # reg.fit(df[["linreg","lasso","exp","expmix"]], df["x1"])
    pred = reg.predict(df[["linreg","lasso","exp"]]) #,"expmix"]])
    print(reg.coef_)
    print(reg.intercept_)
    print(sklearn.metrics.mean_squared_error(df["x1"],pred))
    print(sklearn.metrics.r2_score(pred,df["x1"]))
    raise


def predict_deviations():
    df, _, _ = compute_deviations()
    print(df)
    reg = sklearn.linear_model.LassoCV(normalize=True)
    #df["abs_norm_err"] = abs(df["norm_err"])
    df["yr_sq"] = df["year"] * df["year"]
    regressors = ["exp_yr","exp_yr_sd","edge_sd","2020_sd"]
    pred_var = "abs_err"
    reg.fit(df[regressors], df[pred_var])
    print(df[regressors])
    df["reg"] = reg.predict(df[regressors])
    print(reg.coef_)
    print(reg.intercept_)
    print(sklearn.metrics.mean_squared_error(df[pred_var],df["reg"]))
    print(sklearn.metrics.r2_score(df[pred_var],df["reg"]))

def compute_deviations():
    df = pd.DataFrame()
    event_info = {}
    for event in get_all_events():
        year_to_scores = get_normalized_scores(event)
        year_to_scores = {k:v for k,v in year_to_scores.items() if len(v) > 0}
        years = [y for y in year_to_scores]
        data = pd.DataFrame({"year":years})
        data["event"] = event
        data["x1"] = data.apply(lambda x: year_to_scores[x["year"]][0],axis=1)

        #exponential fit
        exp = ExpModel(max_exp = 0.1)
        exp.fit(data["year"].values, data["x1"].values)

        # print(exp.params)
        data["pred"] = exp.predict(years)
        data["err"] = data["pred"] - data["x1"]
        data["abs_err"] = abs(data["err"])

        wr = max(data["x1"])
        wr_year = data[data["x1"] == wr]["year"].values[0]

        event_info[event] = {
            "event": event,
            "model": exp,
            "record": wr,
            "wr_year": wr_year,
            "sd_err": data["err"].std(),
            "var_err": data["err"].var(),
            "min_yr": min(data["year"]),
            "max_yr": max(data["year"])
        }
        data["sd_err"] = event_info[event]["sd_err"]
        data["min_yr"] = event_info[event]["min_yr"]
        data["max_yr"] = event_info[event]["max_yr"]

        def compute_exp_yr(year):
            return math.e**(-0.005*year)
        data["exp_yr"] = data.apply(lambda x: compute_exp_yr(x["year"]),axis=1)
        data["exp_yr_sd"] = data["exp_yr"] * data["sd_err"]

        def edge_feature(year, min_yr, max_yr):
            border = 4
            if max_yr - year <= border:
                return abs(year - (max_yr-border))
            else:
                return 0
        data["edge"] = data.apply(lambda x: edge_feature(x["year"],x["min_yr"],x["max_yr"]), axis=1)
        data["edge_sd"] = data["edge"] * data["sd_err"]

        data["norm_err"] = data["err"] / data["sd_err"]
        data["2020_sd"] = (data["year"] == 2020) * data["sd_err"]

        def predict_sd_err(year, sd_err, min_yr, max_yr):
            exp_yr = compute_exp_yr(year)
            edge = edge_feature(year, min_yr, max_yr)

            #fit determined by predict_deviations
            #TODO: better to predict abs(err) or err**2?
            #used abs because the Rsq was .18 vs .13 but not sure about that...
            coeffs = [1.81748967e+02, 1.49982883e+04, 4.52570019e-02, 6.16761402e-01]
            coeffs[3] = 0 #don't want to punish good 2020 performances for the high variance that year
            vars_ = [exp_yr, sd_err * exp_yr, edge * sd_err, (year == 2020) * sd_err]
            intercept = -0.007987783579923283
            predicted_abs_err = intercept + sum([a*x for a,x in zip(coeffs, vars_)])

            #ev of abs(normal) is sqrt(2/pi) * sd
            #https://en.wikipedia.org/wiki/Folded_normal_distribution
            return (predicted_abs_err / (2/math.pi)**0.5)


        data["predicted_abs_err"] = data.apply(lambda x: predict_sd_err(x["year"],x["sd_err"],x["min_yr"],x["max_yr"]),axis=1)
        data["norm_err2"] = data["err"] / data["predicted_abs_err"]
        data["performance"] = scipy.stats.norm.cdf(-1 * data["norm_err2"])

        df = df.append(data, ignore_index=True)

        predicted_sd_err = predict_sd_err(
            MAX_YEAR,
            event_info[event]["sd_err"],
            event_info[event]["min_yr"],
            event_info[event]["max_yr"],
        )
        raw_difficulty = (wr - exp.predict([MAX_YEAR])[0]) / predicted_sd_err
        event_info[event]["difficulty"] = scipy.stats.norm.cdf(raw_difficulty)

    events = pd.DataFrame([event_info[e] for e in event_info])
    events = events.sort_values(by="difficulty", ascending=False)
    events.to_csv("/tmp/events.csv",index=False)

    df = df.sort_values(by="norm_err2", ascending=True)
    df.to_csv("/tmp/perfs.csv",index=False)
    print(df)
    return df, event_info, predict_sd_err

def years_to_wr():
    _, event_info, predict_sd_err = compute_deviations()
    for event in event_info:
        prob = 1 #probability that the existing record survives
        for yr in range(MAX_YEAR+1,MAX_YEAR+51):
            pred = event_info[event]["model"].predict([yr])[0]
            record = event_info[event]["record"]
            predicted_sd_err = predict_sd_err(
                yr,
                event_info[event]["sd_err"],
                event_info[event]["min_yr"],
                event_info[event]["max_yr"]
            )
            z_score = (record - pred) / predicted_sd_err
            survival_prob = scipy.stats.norm.cdf(z_score)
            print(event, yr, record, pred, survival_prob)
            # print(record)
            # print(pred)
            # print(predicted_sd)
            # print(z_score)
            prob *= survival_prob
            if prob < 0.5:
                print(event, yr, "done!")
                break

if __name__ == "__main__":
    #years_to_wr()
    predict_deviations()
    raise

    #all swimming
    # for event in ["50freestyle", "100freestyle", "200freestyle", "400freestyle", "800freestyle", "1500freestyle", "100backstroke", "200backstroke", "100breaststroke", "200breaststroke", "100butterfly", "200butterfly", "200medley", "400medley"]:
    #     process_event(event)

    #track
    #process_event('hammer', True)

    #TODO: since predicting the extreme value from the normal directly isn't working
    #(predicts too many records)
    #try to predict the extreme value for each year/event from mean, sd, n
    #tentatively maybe try (max - mean) ~ C * sd * sqrt(log(n))
    #and check if C varies by event? (does this need an intercept?)
    #based on the idea that the max of k draws from standard normal ~ sqrt(log(k))
    #https://stats.stackexchange.com/a/343936
    #could also compute wr impressiveness by finding n that would predict wr for the maximum value
    #(divided by the actual n for that event+year)
    #control is either: max ~ C * t
    #or max ~ A + B * e ** -kt if we can spare another parameter
    #one more thought: in the original fit, instead of assuming the mean exponentially decreases with time
    #what about assuming peak performance decreases exponentially over time? this allows for eg: standards to change
    #in the cutoff for admission as an elite athlete (which could tweak mean+sd substantially) without breaking the regression

    df = pd.DataFrame()
    exp_preds = {}
    for year in range(1960,2015):
        year_preds = {}
        for event in get_all_events():
            data = project_yearly_best(event, year)
            if data:
                df = df.append(data, ignore_index=True)

            #year_preds.update(fit_record_improvement(year, [event]))
        #exp_preds[year] = fit_record_improvement(year)
    # print(exp_preds)
    # print(df)
    # df["exp_pred"] = df.apply(lambda x: exp_preds[x["year"]][x["event"]] - 1, axis=1)
    # print("here")

    reg = sklearn.linear_model.LassoCV(normalize=True) #sklearn.linear_model.LinearRegression()
    df["trend_gap"] = df["trend_1"] - df["trend_8"]
    df["trend_avg"] = (df["trend_1"] + df["trend_2"] + df["trend_4"] + df["trend_8"] + df["trend_16"] + df["trend_32"])/6
    df["gap_diff1"] = df["gap_est_curr"] - df["gap_est_avg"]
    df["gap_diff2"] = df["gap_curr"] - df["gap_avg"]
    df["gap_test"] = df["gap_curr"] - df["gap_est_curr"]
    df["gap_diff3"] = df["gap_est2_curr"] - df["gap_est2_avg"]
    df["x8_delta"] = df["x8_fut"] - df["gap_curr"]
    df["test"] = df["best_fut"] - df["fit_curr"]
    df["swim"] = df.apply(lambda x: 1 * (x["event"] in SWIMMING_EVENTS), axis=1)
    df["field"] = df.apply(lambda x: 1 * (x["event"] in FIELD_EVENTS), axis=1)
    df["swim_trend"] = df["swim"] * df["trend_1"]
    df["field_trend"] = df["field"] * df["trend_1"]
    df["swim_yr_range"] = df["swim"] * df["yr_range"]
    df["field_yr_range"] = df["field"] * df["yr_range"]
    df["swim_yr"] = df["swim"] * df["year"]
    df["field_yr"] = df["field"] * df["year"]
    df["exp_yr"] = df.apply(lambda x: math.e ** ((2020-x["year"])/200), axis=1)
    df["field_exp_yr"] = df["field"] * df["exp_yr"]
    df["fut_by_yr"] = df["best_fut"] / df["exp_yr"]
    regressors = ["trend_1_a","swim","field", "gap_diff2", "fit_curr"] #["trend_1","fit_curr","swim","field","gap_diff2"] #,"swim_trend","field_trend"] #,"trend_8","fit_curr", "gap_diff2", "swim", "field"] #,"fit_curr"] #trend_gap"] #,"exp_pred"]
    pred_var = "best_fut" #"best_fut"
    print(df["trend_avg"].mean())
    #regression question: how fast does 8th best grow?
    #ans: some fixed rate + a % of gap between 8th and 1st as of start of period
    #size of gap explains a lot!
    #question: why doesn't improvement rate in the training set factor in much
    #once gap size is accounted for? interesting.

    #regression question: how fast does the gap between 1st and 8th shrink?
    #on average gap size *increased*, from 2.07% to 2.48%, not sure if significant
    #ans: gap size for each events largely the same from year to year, reverts to mean for that event
    #also gap size relatively shrinking for the events that were improving quicker in training set
    #by about 35% of the improvement rate
    #question: then why isn't the gap size shrinking in general? on average we're seeing ~0.11% improvement per year
    # -> 0.55% over 5 years -> should shrink the gap by 0.19%? hmm maybe that's small compared to the error bars though
    #question: also why isn't there a relationship within the training set between improvement and shrinking gap?

    #regression question: how fast does 1st place grow?
    #ans: the gap between 1st and 8th largely reverts its mean, which varies by event
    #then add a constant plus a fraction of the size of the gap
    #on top of that events that were improving quicker before improve more slowly now???
    #could argue that there's only a certain amount of improvement to be made
    #and so if more is made in period 1 then there's less to be made in period 2
    #but why don't we see that effect when looking at 8th best?
    #need a better explanation
    #how about:
    #future rate of improvement for the field is mostly predicted by gap size
    #future rate of improvement for the best is predicted by improvement for the
    #field - change in gap size
    #change in gap size is larger for categories with faster improvement
    #-> faster improvement *for a given gap size* predicts slower improvement in
    #peak performance going forward
    #there is a decent correlation between gap_2014 and trend_avg (rsq = ~0.3)

    #can't explain the negative relationship between past improvement and
    #future 1st place improvement -> dropping that variable. it doesn't
    #explain too much anyway

    #now to fit all the events:
    #how about

    reg.fit(df[regressors], df[pred_var])
    print(df[[pred_var] + regressors])
    print(reg.coef_)
    print(reg.intercept_)
    df["pred"] = reg.predict(df[regressors])
    df[["event", "year", "pred", pred_var, "raw_fut", "raw_curr", "raw_x8", "fit_curr"] + regressors].to_csv("/tmp/preds.csv", index=False)

    print(sklearn.metrics.mean_squared_error(df[pred_var],df["pred"]))
    print(sklearn.metrics.r2_score(df[pred_var],df["pred"]))

    df["naive_err"] = df.apply(lambda x: -x["best_fut"],axis=1)
    df["pred_err"] = df.apply(lambda x: x["pred"] - x["best_fut"],axis=1)
    print(df)

    # for event in ['100m']: #'100freestyle']: #EVENT_TO_SIGN:
    #     year_to_scores = json.load(open(f'/tmp/records_{event}.json'))
    #     year_to_scores = {int(k):v for k,v in year_to_scores.items()}
    #     # for year in year_to_scores:
    #     #     scores = year_to_scores[year]
    #     #     xmin = min(scores)
    #     #     xmax = max(scores)
    #     #     cnt = len(scores)
    #     #     peak = xmax
    #     #     def normal_cost_fn(theta):
    #     #         loc, scale = theta
    #     #         ratio = scipy.stats.norm.sf(xmin, loc, scale) / scipy.stats.norm.sf(peak, loc, scale)
    #     #         return (math.log(ratio) - math.log(cnt)) ** 2
    #     #     print(xmin)
    #     #     print(xmax)
    #     #     loc_guess = 2*xmin - peak
    #     #     scale_guess = (peak - xmin) / 2 #2sds from xmin to the peak
    #     #     loc, scale = opt.fmin(normal_cost_fn, [loc_guess, scale_guess])
    #     #     print(scipy.stats.norm.sf(xmin, loc, scale) / scipy.stats.norm.sf(peak, loc, scale))
    #     #     print(len(scores))
    #     print(event)
    #     # fit_peak_trend2(year_to_scores)
