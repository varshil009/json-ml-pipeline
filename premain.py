from striprtf.striprtf import rtf_to_text
import json
#==============================================================================================================
from model import model_pipeline
from preprocess import preprocess_pipeline
from dimred import feature_reduction_pipeline


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


def execute(addrs, json = False):
    if not json:
        with open(addrs, "r", encoding="utf-8") as file:
            rtf_content = file.read()
            
        text = rtf_to_text(rtf_content)
        dic = json.loads(text)

    else:
        dic = json.load(open(addrs, 'r', encoding='utf-8'))

    k = preprocess_pipeline(dic)
    u = k.data_selection()

    local_dic = dic["design_state_data"]["feature_reduction"]
    target = dic["design_state_data"]["target"]["target"]
    dimred = feature_reduction_pipeline(local_dic, u[1][0], u[1][1], target)
    k = dimred.execute()

    X_train, X_test = k  
    Y_train = u[1][2]  
    Y_test = u[1][3]   

    run_pipeline(X_train, X_test, Y_train, Y_test, dic)
