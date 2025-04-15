# 🧠 ML Pipeline Automation (Structured from UI JSON)
This project builds a complete, flexible, and modular machine learning pipeline based on a configuration JSON (exported from a UI interface). It supports preprocessing, feature engineering, and model training – all driven by a single `.json` file.
---

## 📁 Project Structure
- **preprocess_pipeline**: Handles reading datasets, imputing missing values, encoding text features (hashing or label encoding), and generating interaction features.
- **feature_reduction_pipeline**: Reduces features using techniques like correlation thresholding, PCA, or tree-based importance.
- **ml_pipeline**: Selects and trains a model (based on JSON input) with `GridSearchCV`, and evaluates it using standard metrics.
- **algoparams_from_ui.json.rtf**: Main JSON file containing all experiment configs (dataset path, target variable, feature interactions, models, hyperparameters, etc.).
---

## 🚀 How It Execute
- Just ensure you have all the dependecies - sklearn, numpy, pandas, striprtf(to handle rich text format), xgboost, json
- clone this repo and in same address run this python file
- OR JUST REPLACE THE ADDRESS IN IPYNB AND RUN

```
from premain import execute

# if you are using rich text format then...
addrs = r"address/to/your/file.rtf"
execute(addrs)

# if directly giving the json file
addrs = r"address/to/your/file.json"
execute(addrs, json = True)
```
