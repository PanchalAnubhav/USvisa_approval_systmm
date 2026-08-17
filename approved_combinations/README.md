# 📋 Approved Visa Combinations — Index

This folder contains verified **Visa Approved** input combinations for each continent, discovered through systematic ML model testing.

Each file lists confirmed combinations that produce an **Approved** prediction from the trained CatBoost model.

---

## Files

| Continent | File | Approved Levels |
|---|---|---|
| 🌏 Asia | [asia.md](./asia.md) | High School, Bachelor's, Master's |
| 🌍 Africa | [africa.md](./africa.md) | High School, Bachelor's, Master's |
| 🌍 Europe | [europe.md](./europe.md) | High School only |
| 🌎 North America | [north_america.md](./north_america.md) | High School, Bachelor's, Master's |
| 🌎 South America | [south_america.md](./south_america.md) | High School, Bachelor's, Master's |
| 🌏 Oceania | [oceania.md](./oceania.md) | High School, Bachelor's |

---

## Key Observations

- **Doctorate gets denied everywhere** — the model learned that Doctorate-level applicants are statistically less likely to be approved (likely reflects dataset distribution, not real-world policy)
- **Europe is heavily biased toward denial** for higher education levels — dataset imbalance issue
- **Job Experience (Y) is almost always required** for approval
- **West and Northeast regions** are the most approval-friendly
- **Wage range $30K–$95K/year** covers most approved cases depending on education level

---

## How to Test

Run the app and enter any combination from these files at [http://localhost:8080](http://localhost:8080)
