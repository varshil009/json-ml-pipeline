from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import time
import numpy as np


class model_pipeline:
    def __init__(self, dic):
        self.dic = dic
        # get design state data part
        self.design_data = self.dic["design_state_data"]
        # get pred type
        self.prediction_type = self.design_data["target"]["prediction_type"]
        # get algo
        self.selected_algorithms = self._get_selected_algorithms()
        self.best_model = None
        self.best_score = float('-inf') if self.prediction_type == "Classification" else float('inf')
        self.best_params = None
        
    def _get_selected_algorithms(self):
        selected = []
        for algo_name, algo_config in self.design_data["algorithms"].items():
            if algo_config["is_selected"]:
                selected.append((algo_name, algo_config))
                #print(algo_name, algo_config)
        return selected
        
    def apply_sample_weights(self, X_train, Y_train):
        weight_strategy = self.design_data["weighting_stratergy"]
        if weight_strategy["weighting_stratergy_method"] == "Sample weights":
            weight_var = weight_strategy["weighting_stratergy_weight_variable"]
            
            if weight_var in X_train.columns:
                sample_weights = X_train[weight_var].values
                # remove weight variable from features
                X_train = X_train.drop(columns=[weight_var])
                return X_train, Y_train, sample_weights
                
        return X_train, Y_train, None
    
    def create_grid_params(self, algo_name, algo_config):
        if algo_name == "RandomForestClassifier" or algo_name == "RandomForestRegressor":
            return  {
                'n_estimators': list(range(algo_config["min_trees"], algo_config["max_trees"] + 1, 5)),
                'max_depth': list(range(algo_config["min_depth"], algo_config["max_depth"] + 1, 5)),
                'min_samples_leaf': list(range(algo_config["min_samples_per_leaf_min_value"], 
                                           algo_config["min_samples_per_leaf_max_value"] + 1, 5))
            }
            
        elif algo_name == "GBTClassifier" or algo_name == "GBTRegressor":
            return {
                'n_estimators': algo_config["num_of_BoostingStages"],
                'max_depth': list(range(algo_config["min_depth"], algo_config["max_depth"] + 1)),
                'learning_rate': [algo_config["min_stepsize"] + i * 0.05 for i in range(
                    int((algo_config["max_stepsize"] - algo_config["min_stepsize"]) / 0.05) + 1)],
                'subsample': [algo_config["min_subsample"] + i * 0.1 for i in range(
                    int((algo_config["max_subsample"] - algo_config["min_subsample"]) / 0.1) + 1)]
            }
            
        elif algo_name == "LinearRegression" or algo_name == "LogisticRegression":
            return {
                'max_iter': list(range(algo_config["min_iter"], algo_config["max_iter"] + 10, 10)),
                'tol': [0.0001, 0.001, 0.01],
                'C': [1.0 / (algo_config["min_regparam"] + i * 0.1) for i in range(
                    int((algo_config["max_regparam"] - algo_config["min_regparam"]) / 0.1) + 1)],  
                'l1_ratio': [algo_config["min_elasticnet"] + i * 0.1 for i in range(
                    int((algo_config["max_elasticnet"] - algo_config["min_elasticnet"]) / 0.1) + 1)] if "elasticnet" in algo_config else [0.5]
            }
            
        elif algo_name == "RidgeRegression":
            return {
                'max_iter': list(range(algo_config["min_iter"], algo_config["max_iter"] + 10, 10)),
                'alpha': [algo_config["min_regparam"] + i * 0.1 for i in range(
                    int((algo_config["max_regparam"] - algo_config["min_regparam"]) / 0.1) + 1)]
            }
            
        elif algo_name == "LassoRegression":
            return {
                'max_iter': list(range(algo_config["min_iter"], algo_config["max_iter"] + 10, 10)),
                'alpha': [algo_config["min_regparam"] + i * 0.1 for i in range(
                    int((algo_config["max_regparam"] - algo_config["min_regparam"]) / 0.1) + 1)]
            }
            
        elif algo_name == "ElasticNetRegression":
            return {
                'max_iter': list(range(algo_config["min_iter"], algo_config["max_iter"] + 10, 10)),
                'alpha': [algo_config["min_regparam"] + i * 0.1 for i in range(
                    int((algo_config["max_regparam"] - algo_config["min_regparam"]) / 0.1) + 1)],
                'l1_ratio': [algo_config["min_elasticnet"] + i * 0.1 for i in range(
                    int((algo_config["max_elasticnet"] - algo_config["min_elasticnet"]) / 0.1) + 1)]
            }
            
        elif algo_name == "xg_boost":
            return {
                'max_depth': list(range(algo_config["max_depth_of_tree"][0], algo_config["max_depth_of_tree"][1] + 1, 10)),
                'learning_rate': algo_config["learningRate"],
                'n_estimators': [10, 20, 50, 100],
                'gamma': algo_config["gamma"] if "gamma" in algo_config else [0],
                'reg_alpha': algo_config["l1_regularization"] if "l1_regularization" in algo_config else [0],
                'reg_lambda': algo_config["l2_regularization"] if "l2_regularization" in algo_config else [1],
                'subsample': algo_config["sub_sample"] if "sub_sample" in algo_config else [0.8],
                'colsample_bytree': algo_config["col_sample_by_tree"] if "col_sample_by_tree" in algo_config else [0.8]
            }
            
        elif algo_name == "DecisionTreeRegressor" or algo_name == "DecisionTreeClassifier":
            return {
                'max_depth': list(range(algo_config["min_depth"], algo_config["max_depth"] + 1)),
                'min_samples_leaf': algo_config["min_samples_per_leaf"],
                'criterion': ['gini' if algo_config.get("use_gini", False) else 'entropy'] if algo_name.endswith("Classifier") else ['squared_error', 'friedman_mse']
            }
            
        elif algo_name == "SVM":
            kernels = []
            if algo_config.get("linear_kernel", False): kernels.append('linear')
            if algo_config.get("polynomial_kernel", False): kernels.append('poly')
            if algo_config.get("sigmoid_kernel", False): kernels.append('sigmoid')
            if algo_config.get("rep_kernel", False): kernels.append('rbf')
            
            gamma_values = ['auto', 'scale']
            if algo_config.get("custom_gamma_values", False):
                gamma_values.extend([0.001, 0.01, 0.1, 1])
                
            return {
                'C': algo_config["c_value"],
                'kernel': kernels if kernels else ['rbf'],
                'gamma': gamma_values,
                'tol': [algo_config["tolerance"] * 0.001],
                'max_iter': [algo_config["max_iterations"] * 100]
            }
            
        elif algo_name == "SGD":
            losses = []
            if algo_config.get("use_logistics", False): losses.append('log_loss')
            if algo_config.get("use_modified_hubber_loss", False): losses.append('modified_huber')
            regularizations = []
            if algo_config.get("use_l1_regularization") == "on": regularizations.append('l1')
            if algo_config.get("use_l2_regularization") == "on": regularizations.append('l2')
            if algo_config.get("use_elastic_net_regularization", False): regularizations.append('elasticnet')
            
            return {
                'loss': losses if losses else ['log_loss'],
                'penalty': regularizations if regularizations else ['l2'],
                'alpha': algo_config["alpha_value"],
                'max_iter': [algo_config["max_iterations"] * 10] if "max_iterations" in algo_config else [1000],
                'tol': [algo_config["tolerance"] * 0.0001] if "tolerance" in algo_config else [0.0001]
            }
            
        elif algo_name == "KNN":
            return {
                'n_neighbors': algo_config["k_value"],
                'weights': ['uniform', 'distance'] if algo_config.get("distance_weighting", False) else ['uniform'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'] if algo_config.get("neighbour_finding_algorithm") == "Automatic" else [algo_config.get("neighbour_finding_algorithm").lower()],
                'p': [algo_config["p_value"]] if "p_value" in algo_config and algo_config["p_value"] > 0 else [2]
            }
            
        elif algo_name == "extra_random_trees":
            return {
                'n_estimators': [x for x in range(algo_config["num_of_trees"][0], algo_config["num_of_trees"][1] + 10, 10)],
                'max_depth': [x for x in range(algo_config["max_depth"][0], algo_config["max_depth"][1] + 1)],
                'min_samples_leaf': algo_config["min_samples_per_leaf"],
                'n_jobs': [algo_config["parallelism"]] if "parallelism" in algo_config else [1]
            }
            
        elif algo_name == "neural_network":
            return {
                'hidden_layer_sizes': [(algo_config["hidden_layer_sizes"][0],), (algo_config["hidden_layer_sizes"][1],)],
                'activation': ['relu', 'tanh', 'logistic'] if not algo_config.get("activation") else [algo_config.get("activation")],
                'alpha': [algo_config["alpha_value"]] if "alpha_value" in algo_config else [0.0001],
                'max_iter': [algo_config["max_iterations"]] if "max_iterations" in algo_config else [200],
                'tol': [algo_config["convergence_tolerance"]] if "convergence_tolerance" in algo_config else [0.0001],
                'learning_rate_init': [algo_config["initial_learning_rate"]] if "initial_learning_rate" in algo_config else [0.001],
                'solver': [algo_config["solver"]] if "solver" in algo_config else ['adam'],
                'batch_size': ['auto'] if algo_config.get("automatic_batching", True) else [200],
                'momentum': [algo_config["momentum"]] if "momentum" in algo_config else [0.9],
                'nesterovs_momentum': [algo_config["use_nesterov_momentum"]] if "use_nesterov_momentum" in algo_config else [True]
            }
            
        else:
            return {}
    
    def get_model(self, algo_name):

        if self.prediction_type.lower() == "classification":
            models = {
                "RandomForestClassifier": RandomForestClassifier(),
                "GBTClassifier": GradientBoostingClassifier(),
                "LogisticRegression": LogisticRegression(solver='saga'),
                "xg_boost": xgb.XGBClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "SVM": SVC(probability=True),
                "SGD": SGDClassifier(),
                "KNN": KNeighborsClassifier(),
                "extra_random_trees": RandomForestClassifier(bootstrap=False, random_state=42),
                "neural_network": MLPClassifier()
            }
            
        else:
            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "LinearRegression": LinearRegression(),
                "RidgeRegression": Ridge(),
                "LassoRegression": Lasso(),
                "GBTRegressor": GradientBoostingRegressor(),
                "xg_boost": xgb.XGBRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "SGD": SGDRegressor(),
                "KNN": KNeighborsRegressor(),
                "extra_random_trees": RandomForestRegressor(bootstrap=False, random_state=42),
                "neural_network": MLPRegressor()
            }

        return models[algo_name]


    def train_and_evaluate(self, X_train, X_test, Y_train, Y_test):

        results = []
        
        X_train, Y_train, sample_weights = self.apply_sample_weights(X_train, Y_train)
        #print(self.selected_algorithms)
        
        for algo_name, algo_config in self.selected_algorithms:
            print(f"Training {algo_name}...")
            model_base = self.get_model(algo_name)
            print(model_base)
            params = self.create_grid_params(algo_name, algo_config)
            
            cv = KFold(n_splits=self.design_data["hyperparameters"].get("num_of_folds", 3), 
                       shuffle=self.design_data["hyperparameters"].get("shuffle_grid", True),
                       random_state=self.design_data["hyperparameters"].get("random_state", 42))
            
            if self.prediction_type.lower() == "classification":
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
            
            grid_search = GridSearchCV(
                estimator=model_base,
                param_grid=params,
                scoring=scoring,
                cv=cv,
                n_jobs=self.design_data["hyperparameters"].get("parallelism", -1),
                verbose=1
            )
            # to get the process time
            start_time = time.time()
            
            if sample_weights is not None:
                grid_search.fit(X_train, Y_train, sample_weight=sample_weights)
            else:
                grid_search.fit(X_train, Y_train)
                
            train_time = time.time() - start_time
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            y_pred = best_model.predict(X_test)
            print(Y_test.shape, y_pred.shape)
            metrics = self.calculate_metrics(Y_test, y_pred)
            metrics['algorithm'] = algo_name
            metrics['train_time'] = train_time
            metrics['best_params'] = best_params
            
            results.append((algo_name, best_model, metrics))
            
            # check type of model to be used get best one
            if self.prediction_type.lower() == "classification":
                if metrics.get('accuracy', 0) > self.best_score:
                    self.best_score = metrics.get('accuracy', 0)
                    self.best_model = best_model
                    self.best_params = best_params
            else:
                if metrics.get('mse', float('inf')) < self.best_score:
                    self.best_score = metrics.get('mse', float('inf'))
                    self.best_model = best_model
                    self.best_params = best_params
        
        return results
    
    def calculate_metrics(self, y_true, y_pred):
        metrics = {}
        
        if self.prediction_type.lower() == "classification":
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_pred)
            except:
                metrics['auc'] = 0
        else:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def predict(self, X_new):
        # predcit from the best
        return self.best_model.predict(X_new)
    
    def get_feature_importance(self):
        if self.best_model is None:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            return self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            return self.best_model.coef_
        else:
            return None
    
def run_pipeline(X_train, X_test, Y_train, Y_test, dic):
    model_pipe = model_pipeline(dic)
    results = model_pipe.train_and_evaluate(X_train, X_test, Y_train, Y_test)
    
    print("\n----- Results Summary -----")
    for algo_name, model, metrics in results:
        print(f"\nAlgorithm: {algo_name}")
        for metric_name, metric_value in metrics.items():
            if metric_name not in ['algorithm', 'train_time', 'best_params']:
                print(f"  {metric_name}: {metric_value}")
        print(f"  Training time: {metrics['train_time']:.2f} seconds")
    
    print(f"\nBest Model: {type(model_pipe.best_model).__name__}")
    print(f"Best Parameters: {model_pipe.best_params}")
    
    if model_pipe.prediction_type.lower() == "classification":
        print(f"Best Accuracy: {model_pipe.best_score:.4f}")
    else:
        print(f"Best MSE: {model_pipe.best_score:.4f}")
    return model_pipe