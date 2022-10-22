# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/psm. For more information about this solution accelerator, visit https://www.databricks.com/blog/2020/10/20/detecting-at-risk-patients-with-real-world-data.html.

# COMMAND ----------

# MAGIC %md
# MAGIC # Propensity Score Matching in Observational Research
# MAGIC 
# MAGIC The purpose of this solution accelerator is to demonstrate how to utilize distributed computing and the lakehouse architecture to perform observational research on longitudinal health records at scale. To showcase an end-to-end scenario, we consider a use-case where we are conducting a study on the effect of a treatment in reducing the rate of hospitalizations among a cohort of COVID-19 patients. 
# MAGIC 
# MAGIC We use a pre-simulated dataset of ~90K covid patients generated using [synthea's covid module](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7531559/). We manually added a cohort of patients exposed to an artifitial treatment, a hypothetical antiviral drug (`databrixovir`), that reduces hospital admissions, which forms the ground truth dataset (where we know the treatment does reduce the probability of admissions). Our aim is to investigate the hypothesis that this treatment is effective in reducing hospital admissions. 
# MAGIC 
# MAGIC ### Cohort creation
# MAGIC 
# MAGIC To simplify the cohort creation process, we have defined a python class `DeltaEHR` that can be used to easily create cohorts on top of [synthea](https://github.com/synthetichealth/synthea) resources (e.g encounters, medications etc). This is done by specifying exclusion/includsion criteria, which is passed as a SQL query string. In addition, cohorts can be created by specifying a window of time relative to a target event (for example being diagnosed with covid) that captures specific event of interest (such as comborbidities, drug exposure etc). For example, the bellow diagram demonstrates the case where we are connstructing a cohort of patients who have been diagnosed with hypertension in the last three years, excluding 10 days leading to their covid diagnosis:
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1O2OmMrJS97FvX1lzrc4xOck5mii1c5cm" width=700>
# MAGIC 
# MAGIC ### Usage:
# MAGIC 
# MAGIC  - Initialize by specifying the path to where synthea delat tables are located. e.g. `my_ehr=DeltaEHR(<path_to_delta>)`
# MAGIC  - Specify list of patients, demographic columns to include and first and last encounter dates for each patinet:
# MAGIC  ```
# MAGIC my_ehr.set_patient_list(demog_list=['BIRTHDATE','MARITAL','RACE','ETHNICITY','GENDER','HEALTHCARE_COVERAGE'])
# MAGIC  ```
# MAGIC  - Add a cohort by just specifying inlcusion/exclusion criteria (as a SQL statement) and the resource (e.g. `medications`, `conditions` etc). For example the following command adds a cohort named `covid` which includes all patients that have been diagnosed for covid (with diagnosis code `840539006`) using synthea's `conditions` dataset.
# MAGIC  
# MAGIC  ```
# MAGIC  my_ehr.add_simple_cohort(cohort_name='covid',resource='conditions',inclusion_criteria='CODE==840539006')
# MAGIC  ```
# MAGIC  - Add a cohort based on historical records, for example exposure to a certain compound within a certain window of time relative to the target event. For example to add a cohort of all patients who have been diagnosed for hypertension (code = `59621000`) based on `conditions` resource, up to three years prior to their covid diagnosis (excluding 10 days leading to their covid diagnosis)
# MAGIC ```
# MAGIC my_ehr.add_cohort(resource='conditions', event_name='hypertension', codes=[59621000],hist_window=3*365,washout_window=10, target_event_name='covid')
# MAGIC ```
# MAGIC 
# MAGIC - Combine all cohorts into a dataset of different features with rows corresponding to a patient in the target population under study, and columns inidicating whether the patient belongs to a given cohort or not. These indicator variables, in combination with additional demographic features (age,gender, ethnicty etc) form the features to be used to conduct the study. For example:
# MAGIC ```
# MAGIC my_dataset=my_ehr.combine_all_cohorts()
# MAGIC display(my_dataset)
# MAGIC ```
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1BPpTWMEV5ZfeksqAVYZuAzzILFEV0Rnb" width=1200>
# MAGIC 
# MAGIC ## Propencity Score Matching
# MAGIC 
# MAGIC To examine the effect of the treatment among the target cohort - cohort of patinents recently diagnosed with covid - we create a dataset of demographic features (age, gender etc), history of comborbidities in the form of binary features (`1` if patinet has been diagnosed with a conidtion in last 3 years and `0` otherwise), and data regarding the outcome status (e.g. being admitted to the hospital or not).
# MAGIC 
# MAGIC To establish any causal relationship between a treatment and outcome we need to match samples based on their propencity of receiving the treatment (for example in case age, ethnicy etc are factors in receiving the treatment). We then proceed to compare the outcome among matched samples.
# MAGIC 
# MAGIC In this package, we use a decision tree calissfier to train a binary calssifer that can predict probability of receiving a treatmnet based on the dataset and use the predicted propabilities as propencity scores. We then examine two methods for propencity score matching:
# MAGIC   1. Startification, which groups samples based on the propencity bucket they belong to,
# MAGIC   2. Nearest neighbor method, which pairs each sample in the treatment group with a sample in the untreated group that has the closest propencity of receiving the treatmnet and then estimate the treatmnet effect based on paired outcomes.
# MAGIC   
# MAGIC For more information see [An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/) for a comprehensive review of the methods used in this notebok.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebooks
# MAGIC  0. `config`: configuring project environment
# MAGIC  1. `data-prep-and-analysis`: Notebook demonstrating cohort creation, exploratory analysis and writing data to delta lake
# MAGIC  2. `psm`: Demonstrating training a binary classifier to estimate propencity scores for a target treatment and to perform propencity score mathcing using two methods of stratification and nearest neighbor matching
# MAGIC  3. `cohort_builder`: cohort creation class based on synthea resources.

# COMMAND ----------

slides_html="""
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vShVibKa-5GQBfUu7dXBosO_rLRxyeiiK6DlkqgCpZWKi4LMI3IksbrMkFOsb_6UodVPQ4xys2ImTeU/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
"""
displayHTML(slides_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## References
# MAGIC | ||
# MAGIC |-----|-----|
# MAGIC |Propensity Score | https://ohdsi.github.io/TheBookOfOhdsi/PopulationLevelEstimation.html#propensity-scores|
# MAGIC |Cohort creation | https://ohdsi.github.io/TheBookOfOhdsi/Cohorts.html|
# MAGIC |An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies|https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/|
# MAGIC |Synthea™ Novel coronavirus (COVID-19) model and synthetic data set|https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483|
