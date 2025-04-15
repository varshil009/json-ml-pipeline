import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd

class preprocess_pipeline():
    def __init__(self, dic):
        self.hashed = []
        self.dic = dic
        
    def read_df(self):
        file = self.dic["design_state_data"]["session_info"]["dataset"]
        df = pd.read_csv(file)
        print("READ THE DATAFRAME")
        return df
        
    def data_selection(self):
        self.df = self.read_df()
        # target vars
        pred = self.dic["design_state_data"]["target"]["prediction_type"]
        self.target = self.dic["design_state_data"]["target"]["target"]
        type_ = self.dic["design_state_data"]["target"]["type"]
        partition = self.dic["design_state_data"]["target"]["partitioning"]

        # train vars
        train_columns = [x for x in self.df.columns if x != self.target]
        train_size = self.dic["design_state_data"]["train"]["train_ratio"]
        
        policy = self.dic["design_state_data"]["train"]["policy"]
        split1 = "random" in self.dic["design_state_data"]["train"]["policy"].lower()
        split2 = "strat" in self.dic["design_state_data"]["train"]["policy"].lower()
        randomness = self.dic["design_state_data"]["train"]["random_seed"]
        KFOLD = self.dic["design_state_data"]["train"]["k_fold"]
        
        # train test split
        if "split" in policy.lower():
            X_train, X_test, Y_train, Y_test = train_test_split(
                                                                self.df[train_columns], 
                                                                self.df[self.target], 
                                                                train_size = train_size, 
                                                                shuffle = split1,
                                                                #stratify = split2,
                                                                random_state = randomness 
                                                                )
            
            X_train, X_test, Y_train, Y_test = self.feature_handling(X_train, X_test, Y_train, Y_test)
            X_train, X_test = self.feature_generation(X_train, X_test)
            print("SPLIT IMPUTATION COMPLETE")
            return "normal_split", [X_train, X_test, Y_train, Y_test]
            
        if KFOLD:
            kf = KFold(n_splits=2, shuffle=True, random_state=42)
            folds = []
            # Generate splits
            for train_idx, test_idx in kf.split(self.df):
                X_train = self.df.loc[train_idx, train_columns]
                X_test = self.df.loc[test_idx, train_columns]
                Y_train = self.df.loc[train_idx, self.target]
                Y_test = self.df.loc[test_idx, self.target]
                
                X_train, X_test, Y_train, Y_test = self.feature_handling(X_train, X_test, Y_train, Y_test)
                X_train, X_test = self.feature_generation(X_train, X_test)
                folds.append([X_train, X_test, Y_train, Y_test])
                
            print("KFOLDED IMPUTATION COMPLETE")
            return "kfold", folds
    
    def feature_handling(self, X_train, X_test, Y_train, Y_test):
        # list features
        features = list(self.dic["design_state_data"]["feature_handling"].keys())
        # grab the feature dictionary part
        features_dic = self.dic["design_state_data"]["feature_handling"]
        #print(features_dic)
        for feature in features:
            if features_dic[feature]["is_selected"]:
                if features_dic[feature]["feature_variable_type"].lower() == "numerical":
                    imp_method = features_dic[feature]["feature_details"]["impute_with"].lower()
                    if imp_method == "custom":
                        imp_value = features_dic[feature]["feature_details"]["impute_value"]
                        if feature == self.target:
                            Y_train.fillna(imp_value,  inplace = True)
                            Y_test.fillna(imp_value,  inplace = True) 
                        else:
                            X_train.fillna({feature : imp_value}, inplace = True)
                            X_test.fillna({feature : imp_value},  inplace = True)
    
                    else:
                        if features_dic[feature]["feature_variable_type"].lower() == "numerical":
                            if "mean" in imp_method or "average" in imp_method:
                                strategy = "mean"
                            elif "median" in imp_method:
                                strategy = "median"
                            elif "mode" in imp_method:
                                strategy = "mode"
                            si = SimpleImputer(strategy = strategy)
    
                            if feature == self.target:
                                si.fit(Y_train)
                                Y_train = si.transform(Y_train)
                                Y_test = si.transform(Y_test)
                            else:
                                si.fit(X_train[feature].values.reshape(-1, 1))
                                X_train[feature] = si.transform(X_train[feature].values.reshape(-1, 1)).ravel()
                                X_test[feature] = si.transform(X_test[feature].values.reshape(-1, 1)).ravel()
                                
                else: # for text data
                    self.hashed.append(feature)
                    strategy = "most_frequent"
    
                    si = SimpleImputer(strategy = strategy)
    
                    if feature == self.target:
                        si.fit(Y_train.values.reshape(-1, 1))
                        Y_train = si.transform(Y_train.values.reshape(-1, 1)).ravel()
                        Y_test = si.transform(Y_test.values.reshape(-1, 1)).ravel()
                    else:
                        si.fit(X_train[feature].values.reshape(-1, 1))
                        X_train[feature] = si.transform(X_train[feature].values.reshape(-1, 1)).ravel()
                        X_test[feature] = si.transform(X_test[feature].values.reshape(-1, 1)).ravel()
                    """
                    not expecting the target variable to be hashed or tokenized so ignoring y_train, 
                    but if it is, we will use Label encoder instead
                    """
                    if feature == self.target:
                        e = LabelEncoder()
                        e.fit(Y_train)
                        Y_train = e.transform(Y_train)
                        Y_test = e.transform(Y_test)
                        
                    else:
                        # use hashing evctorizer
                        hash_columns = features_dic[feature]["feature_details"]["hash_columns"]
                        if hash_columns != 0: 
                            vectorizer = HashingVectorizer(n_features = hash_columns, stop_words = 'english', alternate_sign = False)
            
                            hashed_X_train = vectorizer.transform(X_train[feature].astype(str)).toarray()
                            hashed_X_test = vectorizer.transform(X_test[feature].astype(str)).toarray()
                            
                            # vectorizer returns dense array create columns out of that 
                            hashed_X_train_df = pd.DataFrame(hashed_X_train, index=X_train.index,
                                  columns=[f"hashed_{feature}_{i}" for i in range(hashed_X_train.shape[1])])
                            hashed_X_test_df = pd.DataFrame(hashed_X_test, index=X_test.index,
                                 columns=[f"hashed_{feature}_{i}" for i in range(hashed_X_test.shape[1])])

                            # join
                            X_train = pd.concat([X_train, hashed_X_train_df], axis=1)
                            X_test = pd.concat([X_test, hashed_X_test_df], axis=1)

                            # drop the string col
                            X_train.drop(feature, axis = 1, inplace = True)
                            X_test.drop(feature, axis = 1, inplace = True)
                            
                        else:
                            e = LabelEncoder()
                            e.fit(X_train[feature])
                            X_train[feature] = e.transform(X_train[feature].values.reshape(-1, 1)).flatten()
                            X_test[feature]= e.transform(X_test[feature].values.reshape(-1, 1)).flatten()
                            
        print("Feature handling complete")
        #display(X_test)
        return X_train, X_test, Y_train, Y_test  

    def interactionx(self, interaction_list, interaction_type, X_train, X_test):
        """
        handles interaction and takes care of hashed array interactions if any 
        """
        functiondic = {
                       "linear" : lambda x, y : x + y,
                       "poly": lambda x, y: np.where((y == 0) | pd.isna(y), 0, x / y),
                        "expl": lambda x, y: np.where((y == 0) | pd.isna(y), 0, x / y),

                      }
        
        function = functiondic[interaction_type]
        for x, y in interaction_list:
            # skip when target is given for interaction, this causes dataleakage
            if y == self.target: continue
            if x == self.target: continue
            
            if x in self.hashed:
                cols = [p for p in X_train.columns if f"hashed_{x}" in p]
                for hashed_col in cols:
                    X_train[f"{hashed_col}_l_{y}"] = function(X_train[hashed_col], X_train[y])
                    X_test[f"{hashed_col}_l_{y}"] = function(X_test[hashed_col], X_test[y])

            elif y in self.hashed:
                cols = [p for p in X_train.columns if f"hashed_{y}" in p]
                for hashed_col in cols:
                    X_train[f"{x}_l_{hashed_col}"] = function(X_train[x], X_train[hashed_col])
                    X_test[f"{x}_l_{hashed_col}"] = function(X_test[x], X_test[hashed_col])
            
            # not expecting both to be hashed would be worst case scenario
            elif x in self.hashed and y in self.hashed:
                colx = [p for p in X_train.columns if f"hashed_{x}" in p]
                coly = [p for p in X_train.columns if f"hashed_{y}" in p]
                
                for hashed_x in colx:
                    for hashed_y in coly:
                        X_train[f"{hashed_x}_l_{hashed_y}"] = function(X_train[hashed_x], X_train[hashed_y])
                        X_test[f"{hashed_x}_l_{hashed_y}"] = function(X_test[hashed_x], X_test[hashed_y])
            else:
                X_train[f"{x}_l_{y}"] = function(X_train[x], X_train[y])
                X_test[f"{x}_l_{y}"] = function(X_test[x], X_test[y])
        
        print("INTERACTION COMPLETE")
        return X_train, X_test
    
    def interaction(self, interaction_list, interaction_type, X_train, X_test):
        """
        handles interaction and takes care of hashed array interactions if any 
        with proper NaN handling
        """
        functiondic = {
            "linear": lambda x, y: np.where(pd.isna(x) | pd.isna(y), np.nan, x + y),
            "poly": lambda x, y: np.where(
                (pd.isna(x)) | (pd.isna(y)) | (y == 0), 
                np.nan, 
                x / y
            ),
            "expl": lambda x, y: np.where(
                (pd.isna(x)) | (pd.isna(y)) | (y == 0), 
                np.nan, 
                x / y
            ),
        }
        
        function = functiondic[interaction_type]
        for x, y in interaction_list:
            # skip when target is given for interaction, this causes dataleakage
            if y == self.target or x == self.target: 
                continue
                
            try:
                if x in self.hashed:
                    cols = [p for p in X_train.columns if f"hashed_{x}" in p]
                    for hashed_col in cols:
                        X_train[f"{hashed_col}_l_{y}"] = function(X_train[hashed_col], X_train[y])
                        X_test[f"{hashed_col}_l_{y}"] = function(X_test[hashed_col], X_test[y])

                elif y in self.hashed:
                    cols = [p for p in X_train.columns if f"hashed_{y}" in p]
                    for hashed_col in cols:
                        X_train[f"{x}_l_{hashed_col}"] = function(X_train[x], X_train[hashed_col])
                        X_test[f"{x}_l_{hashed_col}"] = function(X_test[x], X_test[hashed_col])
                
                # not expecting both to be hashed would be worst case scenario
                elif x in self.hashed and y in self.hashed:
                    colx = [p for p in X_train.columns if f"hashed_{x}" in p]
                    coly = [p for p in X_train.columns if f"hashed_{y}" in p]
                    
                    for hashed_x in colx:
                        for hashed_y in coly:
                            X_train[f"{hashed_x}_l_{hashed_y}"] = function(X_train[hashed_x], X_train[hashed_y])
                            X_test[f"{hashed_x}_l_{hashed_y}"] = function(X_test[hashed_x], X_test[hashed_y])
                else:
                    X_train[f"{x}_l_{y}"] = function(X_train[x], X_train[y])
                    X_test[f"{x}_l_{y}"] = function(X_test[x], X_test[y])
            except KeyError as e:
                print(f"Warning: Missing column for interaction - {e}")
                continue
            
        print("INTERACTION COMPLETE")
        return X_train, X_test


    def feature_generation(self, X_train, X_test):
        dic = self.dic["design_state_data"]["feature_generation"]
        
        # add new linear features
        X_train, X_test = self.interaction(dic["linear_interactions"], "linear", X_train, X_test)
            
        # add polynomial features
        polynomial_interactions = [x.split("/") for x in dic["polynomial_interactions"]]
        X_train, X_test = self.interaction(polynomial_interactions, "poly", X_train, X_test)
        
        # add explicit pariwaise reln
        explicit_pairwise_interactions = [x.split("/") for x in dic["explicit_pairwise_interactions"]]
        X_train, X_test = self.interaction(explicit_pairwise_interactions, "expl", X_train, X_test)
        print("FEATURE GENERATION COMPLETE")
        return X_train, X_test