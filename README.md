# track-field-swim

Simple page to compare track field and swimming world records across eras. See https://jasontrigg0.github.io/track-field-swim/

Steps to update the page:
1. update to current year in wr_analysis.py, scrape_swimming.py, scrape_track_and_field.py from 2024 to the current year
2. run scrape_swimming.py, scrape_track_and_field.py and debug scrapers as needed
3. run wr_analysis.py predict_deviations() function
3a. optionally take a look at /tmp/perfs.csv. All performances are normalized such that 1.0 is the best performance in the current year and higher numbers are better. x1 is the year's normalized performance and pred is the prediction for the best performance of the year based on historical trends. Can filter for an event and sort by year to look at the performance progression (x1 column) and how it compares to the trend (pred column)
3b. optionally view events by rate of improvement `less /tmp/perfs.csv | pagg -g event --lam 'return x["x1"].max() - x[x["year"] == 2024]["pred"].values[0]' | psort -c lambda_0 | plook`
4. run generateHtml.js. It will error for any new top records that don't have entries in scores.json. Add the relevant information and rerun
