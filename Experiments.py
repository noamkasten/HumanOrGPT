####################################################################################################
#   This python file consists all the 9 experiments involved in the project.
#   The results of the experiments as performed on Kaggle's hardware are presented on the Readme.md
#   and on the report.
####################################################################################################

from Models.BasedOnTrainer import run_models
import logging
import warnings

API_KEY = "PASTE HERE"
LOG_FORMAT = "################################## {} ##################################"

def rephrased_experiments():
    """
    Training and testing the different Rephrased experiments, the visualization isn't enables, but can be.
    :return: None
    """
    print(LOG_FORMAT.format("Rephrased GPT3 data from paper experiment"))
    print("EXP1: Custom model Optuna 50 trials:")
    run_models("Datasets/Rephrased_GPT3_paper.xlsx", apikey=API_KEY,
               datatype="rephrased", mode=0, model=0, trials=1,
               test_set_path="Datasets/Rephrased_GPT3_testset_paper.xlsx")
    print("EXP2: Reference model:")
    run_models("Datasets/Rephrased_GPT3_paper.xlsx", apikey=API_KEY,
               datatype="rephrased", mode=0, model=1,
               test_set_path="Datasets/Rephrased_GPT3_testset_paper.xlsx")

    print(LOG_FORMAT.format("Rephrased GPT3.5 data created from Openai's API"))
    print("EXP3: Custom model Optuna 50 trials:")
    run_models("Datasets/Rephrased_GPT3.5.xlsx", apikey=API_KEY,
               datatype="rephrased", mode=0, model=0, trials=1,
               test_set_path="Datasets/Rephrased_GPT3.5_testset.xlsx")
    print("EXP4: Reference model:")
    run_models("Datasets/Rephrased_GPT3.5.xlsx", apikey=API_KEY,
               datatype="rephrased", mode=0, model=1,
               test_set_path="Datasets/Rephrased_GPT3.5_testset.xlsx")

def generated_experiments():
    """
    Training and testing the different Generated experiments, the visualization isn't enables, but can be.
    :return:
    """
    print(LOG_FORMAT.format("Generated GPT3 data from paper experiment"))
    print("EXP5: Custom model Optuna 50 trials:")
    run_models("Datasets/Generated_GPT3_paper.xlsx", apikey=API_KEY,
               datatype="generated", mode=0, model=0, trials=1)
    print("EXP6: Reference model:")
    run_models("Datasets/Generated_GPT3_paper.xlsx", apikey=API_KEY,
               datatype="generated", mode=0, model=1)

    print(LOG_FORMAT.format("Generated GPT4 data"))
    print("EXP7: Custom model Optuna 50 trials:")
    run_models("Datasets/Generated_GPT4.xlsx", apikey=API_KEY,
               datatype="generated", mode=0, model=0, trials=1)
    print("EXP8: Reference model:")
    run_models("Datasets/Generated_GPT4.xlsx", apikey=API_KEY,
               datatype="generated", mode=0, model=1)



if __name__ == "__main__":
    # Filter warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")
    rephrased_experiments()
    generated_experiments()