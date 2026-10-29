# FootLens Analytics — Match Intelligence

**Player Injuries and Team Performance Dashboard**

An interactive Streamlit dashboard that explores how player injuries relate to team results in football. It was built for the *Mathematics for AI-II* summative assessment (Scenario 1: Player Injuries and Team Performance Dashboard).

> **Live app:** https://1000405footlens.streamlit.app/
> **Screenshots Link:** https://drive.google.com/drive/folders/1Csk7eZM8xro9LvjypNOTLZiVp06B_j4R?usp=sharing 

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Research questions](#2-research-questions)
3. [Dataset](#3-dataset)
4. [Data preprocessing and feature engineering](#4-data-preprocessing-and-feature-engineering)
5. [Mathematics used](#5-mathematics-used)
6. [Dashboard features](#6-dashboard-features)
7. [Prediction model](#7-prediction-model)
8. [Key findings](#8-key-findings)
9. [Screenshots](#9-screenshots)
10. [Project structure](#10-project-structure)
11. [Integration details (tech stack)](#11-integration-details-tech-stack)
12. [Run locally](#12-run-locally)
13. [Deployment instructions (Streamlit Cloud)](#13-deployment-instructions-streamlit-cloud)
14. [Limitations and responsible interpretation](#14-limitations-and-responsible-interpretation)
15. [Possible improvements](#15-possible-improvements)
16. [Author and submission details](#16-author-and-submission-details)

---

## 1. Project overview

FootLens Analytics is a fictional sports-analytics firm that wants to understand how injuries influence a club's performance, in order to inform training schedules, player rotation and squad planning. This project plays the role of the firm's Junior Sports Data Analyst and delivers a dashboard for technical directors and sports managers.

The app:

- loads a dataset of **656 injury records** (42 raw columns) covering **224 players, 8 clubs and 5 seasons (2019/20 to 2023/24)**;
- cleans the data and derives measures such as injury duration, team performance drop and rating change;
- answers **eight research questions** on dedicated pages with interactive Plotly charts;
- offers extra tools: player lookup, two-club comparison and a Random Forest prediction of the team impact of an injury;
- lets the user filter by season, club, position, player, injury type, impact level and age.

**Important:** all results are *associations in observed matches*, not proof of cause and effect.

## 2. Research questions

Each question is a page in the app and is tied to a measurable variable.

| # | Research question | Measure used |
|---|---|---|
| 1 | Which injury types are associated with the greatest decline in team performance? | Mean *Team Performance Drop* by injury type |
| 2 | How does a team's win/loss record change while a key player is absent? | Wins, draws, losses before vs. while absent |
| 3 | How does an individual player's performance change after returning? | *Rating Change* (after minus before) |
| 4 | Are injuries concentrated in particular months or clubs? | Injury counts by month and by club (heatmap) |
| 5 | Which clubs experience the greatest impact? | Mean drop by club, with volume and duration |
| 6 | To what extent does FIFA rating relate to the performance drop? | Pearson correlation |
| 7 | How does win probability change when a player is absent? | Wins ÷ matches in each window |
| 8 | To what extent is age associated with post-recovery performance change? | Pearson correlation and age-band averages |

## 3. Dataset

- **File:** `player_injuries_impact.csv`
- **Size:** 656 rows × 42 columns (53 columns after feature engineering)
- **Coverage:** 224 unique players, 8 clubs, 5 seasons, 12 positions, injury dates from April 2019 to September 2024

**Column groups**

| Group | Columns |
|---|---|
| Player and club | `Name`, `Team Name`, `Position`, `Age`, `Season`, `FIFA rating` |
| Injury | `Injury`, `Date of Injury`, `Date of return` |
| Before the injury (3 matches) | `Match1..3_before_injury_Result`, `_Opposition`, `_GD`, `_Player_rating` |
| While absent (3 missed matches) | `Match1..3_missed_match_Result`, `_Opposition`, `_GD` |
| After return (3 matches) | `Match1..3_after_injury_Result`, `_Opposition`, `_GD`, `_Player_rating` |

`GD` is the team's goal difference in that match. Results are `win`, `draw` or `lose`.

**Data quality (verified from the CSV)**

- 0 exact duplicate rows; 2 repeated name-and-date pairs exist and are not removed.
- 4,782 cells contain the text `N.A.`; these are treated as missing.
- `Date of Injury` is missing for 83 rows (82 of them are Tottenham) and `Date of return` for 87 rows.
- 2 rows have a return date earlier than the injury date; their duration is set to missing.
- The injury column has 139 raw labels (103 when lower-cased) because of case variants such as "Hamstring injury" (72 records) and "hamstring injury" (23 records).

## 4. Data preprocessing and feature engineering

All cleaning happens in the `prepare()` function of `app.py`:

1. Strip whitespace from column names and text values; treat blanks, `"nan"` and `"none"` as missing.
2. Convert `Age`, `FIFA rating` and `GD` to numbers (invalid values become `NaN`).
3. Convert `Date of Injury` and `Date of return` to datetime.
4. Extract numeric player ratings from the `Player_rating` columns with a regular expression (`rating_num()`).
5. Derive the new columns below.

| Derived column | Formula | Unit / meaning |
|---|---|---|
| `Injury Duration` | `Date of return − Date of Injury` | days; negative values set to missing |
| `Avg GD Before / During / After` | mean of the three match goal differences in each window | goals |
| `Team Performance Drop` | `Avg GD Before − Avg GD During` | goals; positive means the team did worse while the player was absent |
| `Avg Rating Before / After` | mean of the three player ratings in each window | rating points |
| `Rating Change` | `Avg Rating After − Avg Rating Before` | rating points |
| `Injury Month`, `Month Number` | month of `Date of Injury` | calendar month |
| `Impact Level` | `High` if drop > 0.75, `Moderate` if > 0.25, else `Low`; `Unknown` if the drop is missing | a bucket of the drop, **not** a medical severity |

## 5. Mathematics used

- **Mean and grouped means** (`groupby` + `agg`) for averages by injury, club, month and age band.
- **Frequency counts** for injuries by type, club and month.
- **Win rate** = wins ÷ matches with a recorded result, pooled over all matching records.
- **Percentage-point difference** = win rate in one window minus win rate in another. This is not a percentage change.
- **Goal difference** and **performance drop** as defined above.
- **Rating change** = average rating after minus before.
- **Pearson correlation (r)** between FIFA rating and drop, and between age and rating change. The app labels |r| ≥ 0.7 strong, ≥ 0.4 moderate, otherwise weak.
- **Ordinary least squares trendline** on scatter plots (via `statsmodels`).
- **Random Forest regression** evaluated with **MAE** (mean absolute error, in goal-difference units) and **R²** (share of variance explained on held-out data; it is not a percent-correct score).

## 6. Dashboard features

The app has a shared sidebar filter panel and 11 pages in top navigation.

**Sidebar filters:** Season, Club, Position, Player, Injury type, Impact level (multi-select) and an Age slider, plus a *Reset filters* button. A caption shows "Showing X of Y records". Filters apply to the analysis pages; the **Tools** page always uses the full dataset. If filters leave no records, pages show a friendly warning with a reset button instead of crashing.

| Page | What it shows |
|---|---|
| **Home** | Six KPI cards (records, players, clubs, average injury length, average goal-difference drop, win rate while absent), a quick summary, injuries by month and win rate before vs. while absent |
| **Injury impact (RQ1)** | Average drop by injury type (top 12), a bubble chart of frequency vs. drop (bubble size = average days out), and a box plot of time out by injury type |
| **Team absence (RQ2)** | Wins/draws/losses before vs. absent, win-rate bars and average goal difference across the injury period |
| **Player recovery (RQ3)** | Leaderboard of biggest rating improvements, a player lookup and a histogram of rating changes |
| **Injury clusters (RQ4)** | Club-by-month heatmap, injuries-by-month line and top-12 clubs by record count |
| **Club impact (RQ5)** | Average drop by club, two bubble charts and a table of all clubs |
| **FIFA rating (RQ6)** | Scatter with trendline, correlation, and average drop by rating band |
| **Win probability (RQ7)** | Win rate before / during / after and match results by period |
| **Age and recovery (RQ8)** | Scatter with trendline, correlation, and average rating change by age group |
| **Tools** | Player lookup, two-club comparison and the prediction tool with feature importances |
| **Data** | Record counts, a missing-value table, plain-language method notes, a 500-row preview and a **download button for the filtered data (CSV)** |

Most pages end with an "insight" box that states the main takeaway and a caution against reading it as causation.

## 7. Prediction model

The **Tools** page trains a scikit-learn pipeline when it is opened:

- **Target:** `Team Performance Drop` (a number, so this is regression).
- **Features:** `Age`, `FIFA rating`, `Position`, `Injury`, `Team Name`, `Season`, `Avg GD Before`, `Avg Rating Before`.
- **Preprocessing:** median imputation for numeric columns; most-frequent imputation and one-hot encoding for categorical columns.
- **Algorithm:** `RandomForestRegressor` (220 trees, max depth 8, min 3 samples per leaf, `random_state=42`).
- **Split:** 80% train / 20% test, `random_state=42`. With 611 eligible records this gives 488 training and 123 test rows.
- **Test results:** MAE ≈ **1.16** goal-difference units, R² ≈ **0.46**. For comparison, always predicting the training mean gives an MAE of about 1.59 on the same test rows.
- **Output:** a predicted drop, its percentile in the historical data and an impact level (High / Moderate / Low).

> **Read with care.** The target equals `Avg GD Before − Avg GD During`, so `Avg GD Before` is mechanically tied to the answer; its large feature importance (about 51%) is partly by construction. The model uses a single random split and is exploratory only. It is not a medical forecast and not a guarantee about future matches.

## 8. Key findings

All values were recalculated from the CSV using the app's own functions.

- **Win rate:** 40.9% before injury (1,645 matches), 38.7% while absent (1,508) and 42.6% after return (1,442). Exact differences: **−2.25 percentage points** (before → absent) and **+3.99 percentage points** (absent → after).
- **Most common injury label:** "Hamstring injury" with 72 records (95 if the lowercase variant is added).
- **Most represented club:** Newcastle with 120 records.
- **Timing:** December is the busiest month (84 of the 573 dated injuries).
- **Clubs:** Brentford (0.62) and Burnley (0.53) have the largest average drops; the other clubs are near zero. Group sizes differ widely.
- **Correlations are weak:** FIFA rating vs. drop r ≈ 0.03 (n = 611); age vs. rating change r ≈ 0.04 (n = 503).
- **Recovery:** 47% of players with ratings on both sides of the injury had a higher average rating after returning (202 players); the mean change is about −0.01 rating points.

## 9. Screenshots

Add your own screenshots of the deployed app to a `screenshots/` folder and update the file names below.

| Page | Screenshot |
|---|---|
| Home | `![Home](screenshots/home.png)` |
| Filters (sidebar) | `![Filters](screenshots/filters.png)` |
| Injury impact (RQ1) | `![RQ1](screenshots/rq1.png)` |
| Injury clusters (RQ4 heatmap) | `![RQ4](screenshots/rq4.png)` |
| Tools and prediction | `![Tools](screenshots/tools.png)` |
| Data and methodology | `![Data](screenshots/data.png)` |

## 10. Project structure

```
.
├── app.py                      # Streamlit application (data prep, filters, pages, model)
├── player_injuries_impact.csv  # Dataset (must sit next to app.py)
├── requirements.txt            # Python dependencies
├── mascot.png                  # Image shown on the Home page
├── screenshots/                # Screenshots of the deployed app
└── README.md
```

`app.py` is a single file. Main building blocks: `load_data()` (cached CSV load), `prepare()` (cleaning and feature engineering), `sidebar_filters()` / `apply_filters()` (filtering), one function per page (`home`, `rq1` … `rq8`, `tools`, `data_page`) and `fit_model()` (Random Forest pipeline).

## 11. Integration details (tech stack)

| Library | Role in this project |
|---|---|
| Python | Language |
| Streamlit (≥ 1.40) | Web app, multi-page navigation, filters, layout, CSV download |
| pandas (≥ 2.2) | Loading, cleaning, `groupby`/`agg`, pivoting |
| NumPy (≥ 1.26) | Numeric operations and missing values |
| Plotly Express (≥ 5.24) | Interactive charts (bar, line, scatter, box, histogram, heatmap) |
| statsmodels (≥ 0.14) | OLS trendlines on scatter plots |
| scikit-learn | Random Forest pipeline, imputation, one-hot encoding, MAE and R² |

**Data flow:** CSV → `load_data()` → `prepare()` → sidebar filters → page functions → Plotly charts. The Tools page trains the model on the full prepared dataset.

`requirements.txt` must list every package the app imports, including scikit-learn:

```
streamlit>=1.40
pandas>=2.2
numpy>=1.26
plotly>=5.24
statsmodels>=0.14
scikit-learn>=1.4
```



## 12. Deployment instructions (Streamlit Cloud)

1. Push `app.py`, `requirements.txt`, `player_injuries_impact.csv`, `mascot.png` and `README.md` to a public GitHub repository (name it as your course instructions require, for example `IADAI102(Student_id)-studentname`).
2. Give repository access to `ai.assignments@wacpinternational.org` if your submission checklist requires it.
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
4. Click **Deploy an app** (or **New app**).
5. Select your repository and branch, and set the main file path to `app.py`.
6. Click **Deploy** and wait for the build. If the build fails with an import error, check that every imported package is in `requirements.txt`.
7. Copy the live URL into the **Live app** line at the top of this README and into your submission document.


## 13. Possible improvements

- Normalise injury labels (for example, lower-case them) before counting.
- Add a minimum-records filter to the injury-impact ranking and show the sample size on every chart.
- Add a per-player before/during/after timeline and an age vs. performance-drop scatter plot.
- Show the comeback leaderboard as a sortable table.
- Cross-validate the model and test variants that exclude `Avg GD Before`.
- Add automated tests, accessibility checks and user testing with real analysts.

## 14. Author and submission details

| Field | Detail |
|---|---|
| Student full name | `M Raksmi Priyasree` |
| Candidate registration number | `1000405` |
| CRS name | Artificial Intelligence |
| Course name | Mathematics for AI-II |
| School name | `Jain Vidyalaya IB World School` |
| GitHub repository | `<repository URL>` |
| Live web app | `<Streamlit Cloud URL>` |

**Data note:** the dataset was supplied with the assessment brief. This project is for educational use.
