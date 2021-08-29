import numpy as np
import scipy.stats
import scipy.optimize as opt
import sklearn.linear_model
import sklearn.metrics
import pandas as pd
import json
import math

MAX_YEAR = 2021

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

def get_all_events():
    # return SWIMMING_EVENTS
    return list(EVENT_TO_SIGN.keys()) + SWIMMING_EVENTS

def get_normalized_scores(event, min_cnt=1):
    year_to_scores = json.load(open(f'/tmp/records_{event}.json'))
    year_to_scores = {int(k):v for k,v in year_to_scores.items() if len(v) >= min_cnt}

    best_curr = max(year_to_scores[MAX_YEAR])
    year_to_scores = {int(k):[x/abs(best_curr) for x in v] for k,v in year_to_scores.items()}
    #after the above most scores are ~1 or ~-1
    #convert all to ~1 to avoid sign issues
    year_to_scores = {int(k):[2+x if x<0 else x for x in v] for k,v in year_to_scores.items()}

    if event in SWIMMING_EVENTS:
        #exclude the supersuit years: 2008+2009
        year_to_scores = {k:v for k,v in year_to_scores.items() if k not in [2008,2009]}

    return year_to_scores

def regress(df, regressors, target):
    reg = sklearn.linear_model.LassoCV(normalize=True)
    reg.fit(df[regressors], df[target])
    predictions = reg.predict(df[regressors])
    print(reg.coef_)
    print(reg.intercept_)
    print(sklearn.metrics.mean_squared_error(df[target],predictions))
    print(sklearn.metrics.r2_score(df[target],predictions))

def project_yearly_best():
    horizon = 5
    df = pd.DataFrame()
    for event in get_all_events():
        print(event)
        for curr_year in range(1960,MAX_YEAR+1):
            proj = project_event_yearly_best(event, curr_year, horizon)
            if proj is not None:
                df = df.append(proj, ignore_index=True)
    regressors = ["trend_1", "gap_diff", "fit_curr", "swim", "field"]
    target = "x1_fut"
    print(df)
    regress(df, regressors, target)

def project_event_yearly_best(event, curr_year, horizon):
    year_to_scores = get_normalized_scores(event)

    if (curr_year + horizon) not in year_to_scores:
        return None

    x1_fut = year_to_scores[curr_year + horizon][0]

    years = [y for y in year_to_scores if y <= curr_year and len(year_to_scores[y]) >= 8]
    if curr_year not in years:
        return None
    if len(years) < 10: #require 10 years of data to get proper trends
        return None

    df = pd.DataFrame({"year":years})
    df["x1"] = df.apply(lambda x: year_to_scores[x["year"]][0],axis=1)
    df["x8"] = df.apply(lambda x: year_to_scores[x["year"]][7],axis=1)
    df["gap"] = (df["x8"] / df["x1"] - 1)

    #trend
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(df[["year"]], df[f"x1"], df.apply(lambda x: math.e**((x["year"]-2020)/10), axis=1))
    trend_1 = reg.coef_[0]
    fit_curr = reg.predict([[curr_year]])[0] - 1

    gap_curr = df[df["year"] == curr_year]["gap"].values[0]
    gap_avg = df["gap"].mean()
    data = {
        "event": event,
        "x1_fut": x1_fut-1,
        "trend_1": trend_1,
        "swim": 1 * (event in SWIMMING_EVENTS),
        "field": 1 * (event in FIELD_EVENTS),
        "gap_diff": gap_curr - gap_avg,
        "fit_curr": fit_curr
    }
    return data

def regress_yearly_best():
    df = pd.DataFrame()
    for event in get_all_events():
        print(event)
        df = df.append(regress_event_yearly_best(event), ignore_index=True)
    return df


def regress_event_yearly_best(event):
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

def find_best_fit():
    #test linear, lasso linear, exponential, and mixture of two exponentials
    #to see which best fits the data
    #tentatively looks like exponential fits best (while mixture overfits), though
    #mixture broke on some swimming events
    df = regress_yearly_best()
    df.to_csv("/tmp/reg.csv", index=False)
    reg = sklearn.linear_model.LassoCV(normalize=True)
    regress(df, ["linreg"], "x1")
    regress(df, ["lasso"], "x1")
    regress(df, ["exp"], "x1")
    regress(df, ["expmix"], "x1")
    regress(df, ["linreg","lasso","exp"], "x1")

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
            #coeffs = [1.81748967e+02, 1.49982883e+04, 4.52570019e-02, 6.16761402e-01]
            coeffs = [2.77718369e+02, 1.51778547e+04, 5.03994140e-02, 4.90335945e-01]
            coeffs[3] = 0 #don't want to punish good 2020 performances for the high variance that year
            vars_ = [exp_yr, sd_err * exp_yr, edge * sd_err, (year == 2020) * sd_err]
            intercept = -0.01267151382948418
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
            prob *= survival_prob
            if prob < 0.5:
                event_info[event]["next_wr_year"] = yr
                break
    print({e:event_info[e]["next_wr_year"] for e in event_info})

if __name__ == "__main__":
    #what model fits the historical data the best?
    #linear vs lasso vs exponential vs mixture of exp
    #find_best_fit()

    #messing around with regressions to project how
    #the various events will progress over the next 5-10 yrs
    #project_yearly_best()

    #simple model of how long until each event has a new wr
    #years_to_wr()

    #fit with exponential model, then predict the size
    #of the deviations, normalize by those and use
    #the results to see the best performances ever
    #results are written to /tmp/perfs.csv or /tmp/events.csv
    predict_deviations()
