import ray
import pandas as pd
import matplotlib.pyplot as plt

from ray.air.config import ScalingConfig
from ray.train.xgboost import XGBoostTrainer


# Create a preprocessor to scale some columns.
from ray.data.preprocessors import StandardScaler

def train_xgboost_model(num_workers, num_cpus, train_dataset, 
                        valid_dataset, preprocessor):
    trainer = XGBoostTrainer(
        scaling_config=ScalingConfig(
            # Number of workers to use for data parallelism.
            num_workers=num_workers,
            resources_per_worker={'CPU': num_cpus},
            # Whether to use GPU acceleration.
            use_gpu=False,
            # Make sure to leave some CPUs free for Ray Data operations.
            _max_cpu_fraction_per_node=0.9,
    ),
    label_column="target",
    num_boost_round=20,
    params={
        # XGBoost specific params
        "objective": "binary:logistic",
        # "tree_method": "gpu_hist",  # uncomment this to use GPUs.
        "eval_metric": ["logloss", "error"],
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
    preprocessor=preprocessor,
    )
    result = trainer.fit()
    return result

def run_experiment(train_dataset, valid_dataset, preprocessor):
    trial_results = []

    num_workers=1
    for num_cpus in [1,2,3,4,5,6]:
        result = train_xgboost_model(num_workers=num_workers, num_cpus=num_cpus, 
                                    train_dataset=train_dataset, valid_dataset=valid_dataset, 
                                    preprocessor=preprocessor)
        result_dict = result.metrics
        result_dict['num_workers'] = num_workers
        result_dict['num_cpus'] = num_cpus
        trial_results.append(result_dict)
        
    num_workers=2
    for num_cpus in [1,2,3]:
        result = train_xgboost_model(num_workers=num_workers, num_cpus=num_cpus,
                                    train_dataset=train_dataset, valid_dataset=valid_dataset, 
                                    preprocessor=preprocessor)
        result_dict = result.metrics
        result_dict['num_workers'] = num_workers
        result_dict['num_cpus'] = num_cpus
        trial_results.append(result_dict)

    return pd.DataFrame(trial_results) 

def main():
    ray.init()
    # Load data from standard s3 example bucket
    # Repartition the data into 6 partitions to enable parallelization
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv").repartition(6)

    # Split data into train and validation.
    train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

    # Create a test dataset by dropping the target column.
    test_dataset = valid_dataset.drop_columns(cols=["target"])
    
    preprocessor = StandardScaler(columns=["mean radius", "mean texture"])
    
    trial_results = run_experiment(train_dataset=train_dataset, valid_dataset=valid_dataset, 
                                   preprocessor=preprocessor)
    
    plt.rcParams.update({'font.size': 22})

    plt.figure(figsize=(20,10))
    pd.DataFrame(trial_results)[['num_workers','num_cpus', 'time_total_s']].\
    set_index('num_cpus').\
    groupby('num_workers')['time_total_s'].plot(
        legend=True, marker='o',
    )
    plt.ylabel('time (s)')
    plt.legend(['num_workers=1', 'num_workers=2'])
    plt.savefig("xgboost_scaling.png")

    
if __name__ == "__main__":
    main()