
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
class feature_reduction_pipeline:

    def __init__(self, dic, X_train, X_test, target):
        self.dic = dic
        self.method = dic["feature_reduction_method"].lower()
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.target = target

    def execute(self):
        if "correlation" in self.method:
            return self.correlation(0.4)
        
        elif "principal" in self.method:
            return self.pca(self.dic[self.method]["num_features_to_keep"])
        
        elif "tree" in self.method:
            depth = self.dic[self.method]["depth_of_trees"]
            numt = self.dic[self.method]["num_of_trees"]
            n = self.dic[self.method]["num_features_to_keep"]
            return self.tree(num_features_to_keep=n, num_trees=numt, depth=depth)
        
        else:
            return self.X_train, self.X_test
        
    def correlation(self, threshold=0.4):
        combined = pd.concat([self.X_train, self.X_test])
        corr_matrix = combined.corr().abs()
         # Keeps the values where the condition is True and replaces everything else with NaN.                  
        """                 |    # makes it triangle # creates a corelation matrix like array with ones
                            |           |                |
                            v           v                v                 """
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = []
        
        for column in upper.columns:
            if any(upper[column] >= threshold):
                to_drop.append(column)
                
        keep = [column for column in corr_matrix.columns if column not in to_drop]
        
        return self.X_train[keep], self.X_test[keep]
    
    def pca(self, n=3):
        combined = pd.concat([self.X_train, self.X_test])
        
        pca = PCA(n_components=n)
        pca.fit(combined)
        
        train_transformed = pca.transform(self.X_train)
        test_transformed = pca.transform(self.X_test)
        
        cols = [f'PC{i+1}' for i in range(n)]
        train_pca = pd.DataFrame(train_transformed, index = self.X_train.index, columns = cols)
        test_pca = pd.DataFrame(test_transformed, index = self.X_test.index, columns = cols)
        
        return train_pca, test_pca
    
    def tree(self, num_features_to_keep=10, num_trees=100, depth=6):
        
        if self.target in self.X_train.columns:
            X = self.X_train.drop(columns=[self.target])
            y = self.X_train[self.target]
        else:
            X = self.X_train
        
        rf = RandomForestClassifier(n_estimators = num_trees, max_depth = depth)
        rf.fit(X, y)
        
        # feature importances
        importances = rf.feature_importances_
        feature_names = X.columns.tolist()
        
        # sort features on importance
        features_with_importance = list(zip(feature_names, importances))
        sorted_features = sorted(features_with_importance, key=lambda x: x[1], reverse=True)
        
        # select top N features
        selected_features = [f[0] for f in sorted_features[:num_features_to_keep]]
        
        X_train_reduced = self.X_train[selected_features]
        X_test_reduced = self.X_test[selected_features]
        
        return X_train_reduced, X_test_reduced