const { readCsvFiles } = require("./util.js");
const { scoreInfo } = require("./scores.js");
const { eventInfo } = require("./events.js");
moment = require('moment');
fs = require('fs');

function generateTabRow(tabInfo) {
  const START = `
      <div class="mdc-tab-bar" role="tablist">
        <div class="mdc-tab-scroller">
          <div class="mdc-tab-scroller__scroll-area">
            <div class="mdc-tab-scroller__scroll-content">
  `;

  let allTabs = [];
  for (let tab of tabInfo) {
    allTabs.push(`

nn                <button class="mdc-tab${tab["active"] ? " mdc-tab--active" : ""}" role="tab" aria-selected="true" tabindex="0">
                  <span class="mdc-tab__content">
                    <span class="mdc-tab__icon material-icons" aria-hidden="true">${tab["icon"]}</span>
                    <span class="mdc-tab__text-label">${tab["label"]}</span>
                  </span>
                  <span class="mdc-tab-indicator${tab["active"] ? " mdc-tab-indicator--active" : ""}">
                    <span class="mdc-tab-indicator__content mdc-tab-indicator__content--underline"></span>
                  </span>
                  <span class="mdc-tab__ripple"></span>
                </button>
    `);
  }

  const END = `
            </div>
          </div>
        </div>
      </div>
  `;

  return START + allTabs.join("\n") + END;
}

function generateHtml(tabInfo, cardInfo) {
  // const now = moment().format("h:mm a, MMMM Do, YYYY");

  const tabHtml = "";
  if (tabInfo.length > 1) {
    tabHtml = generateTabRow(tabInfo);
  }

  let cardHtml = '';
  for (let tab of tabInfo) {
    //tab-panel
    cardHtml += `<div style="margin-top: 25px; flex-direction: column; justify-content: space-around; align-items: center" class="tab-panel ${tab["active"] ? "active" : ""}">`;
    //header
    cardHtml += `<div><h2 style="padding-top: 10px; text-align: center">${tab["title"]}</h2><div style="text-align: center; padding-bottom: 20px; max-width: 1000px">${tab["detail"]}</div></div>\n`;
    //cards
    cardHtml += `<div style="display: flex; margin-top: 25px; flex-direction: row; justify-content: space-around; min-width: 1200px;">`;

    cardHtml += generateCards("All-Time Performances",cardInfo["alltime"]);
    cardHtml += generateCards("Hardest WRs to break today",cardInfo["live"]);
    cardHtml += '</div>';
    cardHtml += '</div>';
  }

  const HTML_HEADER = `
  <head>
    <!-- Required styles for MDC Web -->
    <link rel="stylesheet" href="https://unpkg.com/material-components-web@latest/dist/material-components-web.min.css">
    <link rel="stylesheet" href="mdc-demo-card.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
    <style>
     .tab-panel {
       display: none;
     }
     .tab-panel.active {
       display: flex;
     }
    </style>
  </head>
  <body>
  `;

  const HTML_FOOTER = `
    <!-- Required MDC Web JavaScript library -->
    <script src="https://unpkg.com/material-components-web@latest/dist/material-components-web.min.js"></script>
    <script>
     //setup tabs
     window.onload = function() {
       for (const e of document.querySelectorAll(".mdc-tab-bar")) {
         let tab = new mdc.tabBar.MDCTabBar(e)
         tab.preventDefaultOnClick = true

         tab.listen("MDCTabBar:activated", function({detail: {index: index}}) {
           // Hide all panels.
           for (const t of document.querySelectorAll(".tab-panel")) {
             t.classList.remove("active")
           }

           // Show the current one.
           let tab = document.querySelector(".tab-panel:nth-child(" + (index + 2) + ")")
           tab.classList.add("active")
         })
       }
     };
    </script>
  </body>
</html>
  `;

  return HTML_HEADER + tabHtml + cardHtml + HTML_FOOTER;
}

function generateCard(image, header1, header2, header3, header4) {
    return `
    <div class="mdc-card" style="margin-bottom: 20px; max-width: 500px;">
      <div style="display: flex; justify-content: space-between; align-items: center; min-height: 155px">
        <div style="display: flex; margin-left: 25px; height: 120px; width: 120px; justify-content: center; align-items: center">
          <img style="max-height: 120px; max-width: 100px" src="${image}"></img>
        </div>
        <div>
          <div class="demo-card__primary">
            <h2 style="text-align: right" class="demo-card__title mdc-typography mdc-typography--headline6">${header1}</h2>
            <h2 style="text-align: right" class="demo-card__title mdc-typography mdc-typography--headline6">${header2}</h2>
          </div>
          <div style="text-align: right; padding-bottom: 0" class="demo-card__secondary mdc-typography mdc-typography--body2">${header3}</div>
          <div style="text-align: right" class="demo-card__secondary mdc-typography mdc-typography--body2">${header4}</div>
        </div>
      </div>
    </div>`;
}

function generateCards(title, info) {
  let allCards = [];
  for (let row of info) {
    let card = generateCard(
      row["image"],
      row["header1"],
      row["header2"],
      row["header3"],
      row["header4"]
    );
    allCards.push(card);
  }
  let html = "";
  let header = `<div style="text-align: center; padding-bottom: 10px">${title}</div>`;

  return `  <div style="max-width: 500px">` + header + allCards.join("\n") + "\n" + `  </div>`;
}

function format_performance(x) {
  //x: float from 0-1
  if (x > 0.995) {
    return (x*100).toFixed(2);
  } else {
    return (x*100).toFixed(1);
  }
}

async function main() {
  let perf = [];
  for await (let row of readCsvFiles(['/tmp/perfs.csv'])) {
    perf.push(row);
  }

  let events = [];
  for await (let row of readCsvFiles(['/tmp/events.csv'])) {
    events.push(row);
  }

  let cardInfo = {
    "alltime": [],
    "live": []
  };

  cardInfo["alltime"] = perf.slice(0,50).map((x,i) => {
    const key = x["event"] + "," + x["year"];
    const info = scoreInfo[key];
    if (!info) {
      console.log(x);
      throw new Error(`can't find ${key} in scores.js`);
    }
    return {
      image: `${info["image"]}`,
      header1: `#${i+1} ${info["name"]}`,
      header2: `${format_performance(x["performance"])}`,
      header3: `${eventInfo[x["event"]]}: <a href="${info["link"]}">${info["score"]}</a>`,
      header4: `${info["date"]}`
    }
  });

  cardInfo["live"] = events.map((x,i) => {
    const key = x["event"] + "," + x["wr_year"];
    const info = scoreInfo[key];
    if (!info) {
      console.log(x);
      throw new Error(`can't find ${key} in scores.js`);
    }
    return {
      image: `${info["image"]}`,
      header1: `#${i+1} ${eventInfo[x["event"]]}`,
      header2: `${format_performance(x["difficulty"])}`,
      header3: `<a href="${info["link"]}">${info["score"]}</a> by ${info["name"]}`,
      header4: `${info["date"]}`
    }
  });

  const tabInfo = [
    {
      label: "T&F",
      icon: "calendar_today",
      title: "Top Track, Field, and Swimming Performances",
      detail: `Performances measured by how improbable they were at the time. A score above 50 comes once every year or two, while a score over 99 is a once in a century event. The right column shows the current world records in each event sorted by the performance needed to break them.`,
      active: true
    },
  ];

  const html = generateHtml(tabInfo, cardInfo);

  fs.writeFile('index.html', html, function (err) {
    if (err) return console.log(err);
  });
}

main();
