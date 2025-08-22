# Kaggle Titanic - Machine Learning from Disaster

This repository contains my experiments, feature engineering steps, and models for the classic Kaggle competition: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).

## Current Leaderboard Result

<!-- SECTION START: LEADERBOARD -->
My best public leaderboard score so far:

![Kaggle Titanic Leaderboard Result (Score 0.77990)]("C:\Users\ASUS\OneDrive\Pictures\Screenshots 1\Screenshot 2025-08-22 221036.png")

- Public LB Score: **0.77990**
- Position (public LB at time of upload): (see screenshot)
- Submissions so far: 10
- Continuous improvement over earlier score (e.g., 0.77272 ➜ 0.77990)
<!-- SECTION END: LEADERBOARD -->

## Approach Summary

1. Data Cleaning
   - Filled `Age` with median stratified by `Pclass` and `Sex`
   - Imputed `Embarked` with mode; dropped `Cabin` (high sparsity) or extracted `CabinDeck` (optional branch)

2. Feature Engineering
   - `Title` extracted from `Name` (Mr, Miss, etc.)
   - `FamilySize = SibSp + Parch + 1`
   - `IsAlone = (FamilySize == 1)`
   - Binned / scaled `Fare` (log1p + quantile bin)
   - `TicketGroupSize` (count passengers sharing ticket)
   - (Optional) `CabinDeck` first letter where available

3. Modeling
   - Tried baseline: Logistic Regression
   - Improved with: RandomForest, XGBoost / LightGBM
   - Final blend or tuned model: (e.g.) LightGBM with Bayesian hyperparameter optimization
   - Feature importance used to prune low-impact engineered variables

4. Validation Strategy
   - Stratified K-Fold (k=5) on `Survived`
   - Monitored gap between CV mean and public LB to detect overfitting

5. Submission Generation
   - Used best CV model (or ensemble average)
   - Exported `PassengerId,Survived`

## Next Improvement Ideas

- Add stacking (LogReg meta model on top of tree models + GBDTs)
- Try CatBoost with categorical handling
- Optimize Title grouping (combine rare titles)
- Use cross-validation target encoding for `Ticket`, `Surname`
- Calibrate probabilities (isotonic / Platt) and evaluate log-loss offline

## How to Reproduce

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# (Place raw train.csv / test.csv in data/raw/ or specify path)
python src/train.py
python src/predict.py --model artifacts/best_model.pkl --output submissions/submission_<date>.csv
```

## Environment & Dependencies

See `requirements.txt` (example minimal):

```text
pandas
numpy
scikit-learn
lightgbm
xgboost
jupyter
```

## License

MIT (adjust if different).

## Acknowledgements

Kaggle community discussion threads and starter notebooks for initial inspiration.
