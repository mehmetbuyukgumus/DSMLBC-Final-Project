from sklearn.ensemble import RandomForestClassifier

knn_params = {
            # 'algorithm': ['auto'],
            'leaf_size': range(2,5),
            # 'metric': ['minkowski'],
            'metric_params': [{'p': p} for p in range(1, 4)],
            'n_jobs': [None, 1, -1, 2, 4],
            'n_neighbors': range(2,5),
            'p': range(2,5),
            # 'weights': ['uniform', "distance"]
            }

decision_tree_params = {
                        'ccp_alpha': [0.0, 0.01],
                        'class_weight': ["balanced", "balanced_subsample"],
                        'criterion': ["gini", "entropy", "log_loss"],
                        'max_depth': [None,10,20],
                        'max_features': ["sqrt", "log2"],
                        # 'max_leaf_nodes': None,
                        # 'min_impurity_decrease': [0.0],
                        'min_samples_leaf': [1,2,3],
                        'min_samples_split': [2,4,6],
                        'min_weight_fraction_leaf': [0.0, 0.01, 0.001],
                        # 'monotonic_cst': None,
                        'random_state': [None,42],
                        'splitter': ['best', "random"]
                        }

one_vs_params = {
                'estimator': [RandomForestClassifier()],
                'estimator__max_depth': [50,100,200],
                'estimator__min_samples_split': [2,5,10],
                'estimator__min_samples_leaf': [1,2,3],
                'estimator__max_features': ['sqrt', "log2"],
                'estimator__class_weight': ["balanced", "balanced_subsample"],
                'estimator__criterion': ['gini', "entropy"],
                'estimator__random_state': [None, 42],
                # 'estimator__bootstrap': True,
                # 'estimator__ccp_alpha': 0.0,
                # 'estimator__max_leaf_nodes': None,
                # 'estimator__max_samples': None,
                # 'estimator__min_impurity_decrease': 0.0,
                # 'estimator__min_weight_fraction_leaf': 0.0,
                # 'estimator__monotonic_cst': None,
                # 'estimator__n_estimators': 100,
                # 'estimator__n_jobs': None,
                # 'estimator__oob_score': False,
                # 'estimator__verbose': 0,
                # 'estimator__warm_start': False,
                # 'n_jobs': None
                }


SVC_params ={          
            "C": [0.1, 1],  
            # "gamma": ["scale", "auto", 0.1, 1],
            # "kernel": ["linear", "rbf", "poly"],
            # "class_weight": [None, "balanced"], 
            "degree": [2, 3],  
            "coef0": [0.0, 0.1],
            "max_iter": [1000, 2000],
            "tol": [0.001, 0.01],
            "cache_size": [100, 200],
            # "probability": [True, False]
            # "shrinking": [True, False],
            # "break_ties": [True, False],
            # "verbose": [True, False]
            }


