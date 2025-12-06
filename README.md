# Video Games Market Analytics Project

A comprehensive data analysis project examining video game market data to derive actionable insights for indie developers, game studios, and gamers.

## 📊 Project Overview

This project analyzes a comprehensive dataset of video games across multiple platforms, exploring patterns in pricing, player engagement, profitability, and market trends to provide data-driven recommendations.

## 🎯 Target Audiences & Key Questions

### For Indie Developers
- Which console/platform should we target?
- What genre performs best for indie games?
- Should we go free-to-play or paid?
- What's the optimal release timing?

### For Game Studios
- Should we publish on all platforms simultaneously or wait?
- Should we sign exclusivity contracts with publishers?
- What's the best time to release a video game?

### For Gamers
- How does purchasing expensive AAA titles impact overall game pricing trends?
- Price trend analysis and market dynamics

## 📁 Dataset Description

The dataset includes video game data from multiple platforms and sources:

### Key Metrics
- **Sales & Engagement**: Estimated owners, price, peak concurrent users, playtime statistics
- **Feedback & Ratings**: User reviews (positive/negative), Metacritic scores, recommendations
- **Game Attributes**: Genres, tags, categories, DLC count, achievements
- **Platform Support**: Windows, Mac, Linux compatibility
- **Release Information**: Release dates, developers, publishers

## 🔧 Methodology

### Data Cleaning
1. **Dimensionality Reduction**: Removed administrative metadata irrelevant to statistical analysis
2. **Data Transformation**: 
   - Converted estimated owner ranges to numeric averages
   - Parsed release dates to datetime objects
3. **Missing Value Treatment**:
   - Identified high-impact games with missing data
   - Manual verification and patching from SteamDB and official Steam Store
   - Imputed categorical variables with "Unknown" to preserve records

### Feature Engineering
- **Game Age**: Years since release
- **Steam Score**: Positive/(Positive + Negative) sentiment ratio
- **Is Free**: Binary classification for business model analysis
- **Tags Count**: Number of community tags
- **Indie Classification**: Boolean flag based on genres/tags
- **Estimated Profit**: Price × Estimated Owners

## 📈 Key Findings

### Market Composition
- Indie games represent **~67%** of the catalog (2× non-indie games)
- Despite quantity advantage, indie games average significantly lower concurrent users

### Pricing Insights
- Majority of games priced under $25
- Average game prices fluctuate year-over-year
- Free games distribution similar across indie/non-indie categories
- Indie games tend toward paid models more frequently

### Profitability Analysis
- Non-indie games generate higher average profit per title
- Top publishers: Valve, Rockstar Games, Bethesda Softworks
- Average profit per game varies significantly by publisher

### Player Engagement
- Top genres by playtime: Massively Multiplayer, Free to Play, Early Access
- Peak concurrent users strongly correlate with game type (indie vs. non-indie)

## 🔍 Additional Analyses

### Extras
- Evolution of gaming culture over the years
- Impact of DLC, bug fixes, and patches on player retention
- Platform exclusivity effects on sales

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development

## 📂 Project Structure

```
Video-Games-Market-Analytics/
├── data/
│   └── mainDS.parquet          # Primary dataset
├── notebooks/
│   └── DV_Project(1).ipynb     # Main analysis notebook
├── visualizations/              # Generated charts and graphs
├── reports/                     # Analysis reports and findings
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy seaborn matplotlib
```

### Running the Analysis
1. Clone the repository
2. Ensure `data/mainDS.parquet` is in the data directory
3. Open `notebooks/DV_Project(1).ipynb` in Jupyter
4. Run cells sequentially

## 📊 Data Sources

- **Primary Dataset**: Video Games Market Dataset (Parquet format)
- **Release Date Verification**: Multiple platform sources including SteamDB, Official Store pages

## 👥 Contributors

[Add team member names here]

## 📝 License

[Add license information]

## 🔗 References

- [Metacritic](https://www.metacritic.com/)
- Multiple gaming platform data sources

---

**Note**: This is an academic/research project. Estimated profit calculations are approximations based on available data and do not represent actual revenue figures.
