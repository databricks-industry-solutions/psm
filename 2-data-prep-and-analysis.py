# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/psm. For more information about this solution accelerator, visit https://www.databricks.com/blog/2020/10/20/detecting-at-risk-patients-with-real-world-data.html.

# COMMAND ----------

# MAGIC %md
# MAGIC # Data preparation and QC
# MAGIC In this notebook we start by creating cohorts based on our study design using synthea resources that are already loaded into delta (using `1-data-ingest` notebook).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Initial Setup
# MAGIC First we run `./0-config` notebook to configure our project's environment by setting up the base path, deltalake path, mlflow experiment etc. We then run `./cohort_builder` notebook to import `DeltaEHR` class that is designed to make it easy to create cohorts based on synthea resources.

# COMMAND ----------

project_name='psm'

# COMMAND ----------

# DBTITLE 1,run to access cohort_builder class
# MAGIC %run ./cohort-builder

# COMMAND ----------

# DBTITLE 1,read configs
from pprint import pprint
with open(f'/tmp/{project_name}_configs.json','r') as f:
    settings = json.load(f)
    delta_path = settings['delta_path']
pprint(settings)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we specify our experiment's main parameters, namely: target medication (intervention under study), target event (event defining cohort entry date), and the target outcome (outcome under study)

# COMMAND ----------

# DBTITLE 1,define parameters
target_params = {
  # set the target drug med code
  'target_med_code':20134224, # databrixovir  
  # set the target drug name
  'target_med_name':'databrixovir',
  # set the target event code
  'target_event_code':840539006,
  # set the target event name
  'target_event_name':'covid',
  # set the target outcome
  'target_outcome' : 'admission',
  'target_outcome_code': 1505002
}

# COMMAND ----------

# MAGIC %md
# MAGIC We also would like to include information regarding past histroy of comorbidities (for example obesity etc) that can be later used in our propencity score matching

# COMMAND ----------

# DBTITLE 1,define comorbidities
comorbidities = {
'obesity':[162864005,408512008],
'hypertension':[59621000],
'heart_disease':[53741008], 
'diabetes':[44054006],
'smoking':[449868002],
}

# COMMAND ----------

# DBTITLE 1,log parameteres
import mlflow
mlflow.set_experiment(settings['experiment_name'])
with mlflow.start_run(run_name='cohort-creation'):
  mlflow.log_params({**comorbidities, **target_params})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. cohort creation
# MAGIC To make cohort creation and ETL easier, we created a class `DeltaEHR` that makes data pre-processing easier. Using the available methods we create target and outcome cohorts, based on our inclusion/exclusion criteria. First we start by adding our target cohort (patients diagnosed with covid) and outcome cohort (patients who have been admitted to the hospital). Eeach cohort is then automatically added to the collection of cohorts. Each chort contains three columns, patinet id (`PATIENT`), cohort start index date (`START`) and cohort exit date (`STOP`).

# COMMAND ----------

# DBTITLE 1,add cohorts
delta_ehr=DeltaEHR(delta_path)
delta_ehr.add_simple_cohort(cohort_name='covid',resource='conditions',inclusion_criteria=f"CODE=={target_params['target_event_code']}")
delta_ehr.add_simple_cohort(cohort_name='admission',resource='encounters',inclusion_criteria=f"REASONCODE == {target_params['target_event_code']} AND CODE == {target_params['target_outcome_code']}")
delta_ehr.add_simple_cohort(cohort_name='deceased',resource='patients',inclusion_criteria="DEATHDATE is not null",start_col='DEATHDATE',stop_col='DEATHDATE')
target_cohort=delta_ehr.cohorts['covid']

# COMMAND ----------

# MAGIC %md
# MAGIC next we specify which demographic information to inlclude in the dataset

# COMMAND ----------

# DBTITLE 1,specify demographic features to include
delta_ehr.set_patient_list(demog_list=['BIRTHDATE','MARITAL','RACE','ETHNICITY','GENDER'])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now we add cohorts based on prior events (comorbidities, drug exposure etc). For each comorbid condition of interest, we choose a window of time 
# MAGIC to go back and look for any record of diagnosis of, or exposure to, of a condition of interest. This is done simply by using the `add_cohort` method 
# MAGIC defined in `DeltaEHR` class. This method also allows you to specify a washout window (`gate_window`) as a buffer in cases where we want to ensure effects of treatments do not interfere. For example if this is specified to
# MAGIC 10 days, then if there is an instance of a comorbidity diagnosis within 10 dyas of the target ourcome, we do not inlcude that event. Note that
# MAGIC if you speciy a negative value for the washout window you can include evnets occuring after the target event (see below)
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1O2OmMrJS97FvX1lzrc4xOck5mii1c5cm" width=700>

# COMMAND ----------

# DBTITLE 1,add comorbidity cohorts
for event,codes in list(comorbidities.items()):
  delta_ehr.add_cohort('conditions', event, codes,3*365,10, 'covid')

# COMMAND ----------

# MAGIC %md
# MAGIC Now we add the cohort of patinets that have received the target treatment with 10 days of being diagnosed with covid. This is done the same way as adding cohorts based on historic events, with the difference that in this case we set `hist_winodw=0` and `gate_window=-10`

# COMMAND ----------

# DBTITLE 1,add treatment cohort
delta_ehr.add_cohort('medications', target_params['target_med_name'], target_params['target_med_code'], 0,-10, 'covid')

# COMMAND ----------

# MAGIC %md
# MAGIC Optionally you can also add cohorts correspodning to other treatments, for example:
# MAGIC 
# MAGIC ```
# MAGIC meds_test=(
# MAGIC   delta_ehr.tables['medications'].filter("to_date(START) > to_date('2020-01 01')")
# MAGIC   .join(
# MAGIC     delta_ehr.cohorts['covid'].select('PATIENT'),on='PATIENT')
# MAGIC   .join(delta_ehr.cohorts['admission'].select('PATIENT'),on='PATIENT')
# MAGIC   .groupBy('CODE','DESCRIPTION')
# MAGIC   .count()
# MAGIC   .orderBy(desc('count'))
# MAGIC   .limit(20)
# MAGIC   .collect()
# MAGIC )
# MAGIC 
# MAGIC medications={f"m_{m['CODE']}":m['CODE'] for m in meds_test}
# MAGIC for med,codes in list(medications.items()):
# MAGIC   delta_ehr.add_cohort('medications', med, codes,0,-10, 'covid')
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC we can also add cohort of patients experiencing other symptoms such as blood clots within 20 days of diagnosis with covid

# COMMAND ----------

blood_clot={"blood_clot":234466008}
delta_ehr.add_cohort('conditions','blood_clot',234466008, 0, -20,'covid')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. create dataset
# MAGIC Now we creat the final dataset for our downstream analysis. One of the methods in `DeltaEHR` is `get_cohort_tags()`. This method combines all cohort information in form of columns of indicator functions (cohort membership indicator) and the cohort index date correspoding to each patient id.

# COMMAND ----------

# DBTITLE 1,assemble Cohort Dataset
data_df=delta_ehr.combine_all_cohorts()
data_df.createOrReplaceTempView("delta_ehr_cohorts")
display(data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Exploratory Analysis
# MAGIC Now let's take a look at the dataset and look at trends, such as number of individuals diagnosed with the target condition (covid) over time, and demographic trends and other statistics of interest.

# COMMAND ----------

# DBTITLE 1,number of covid patients
covid_counts_by_age_df= sql("""
  SELECT covid_START, 20*cast(age_at_covid/20 as integer) as age_band, count(*) as count
  FROM delta_ehr_cohorts
  WHERE is_covid == 1
  group by 1, 2
  order by covid_START
""")
display(covid_counts_by_age_df)

# COMMAND ----------

# DBTITLE 1,infection wave by age group
import plotly.express as px
df = covid_counts_by_age_df.toPandas()
fig = px.bar(df, x="covid_START", color='age_band',barmode='stack',y="count")
fig.show()

# COMMAND ----------

# DBTITLE 1,hypertension frequency by race
# MAGIC %sql
# MAGIC SELECT race, avg(is_hypertension) as hypertension_frequency
# MAGIC FROM delta_ehr_cohorts
# MAGIC GROUP BY race
# MAGIC order by 2

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's take a look at some of the trends regarding reported blood clot occurances, such as dirtribution of blood clots among covid patients vs others, distribution of the timeframe within which a blood clot is reported, and look at demographic patterns that may be correlated with these timeframes.

# COMMAND ----------

# DBTITLE 0,blood clot
# MAGIC %sql
# MAGIC SELECT is_covid,
# MAGIC        sum(is_blood_clot) as count
# MAGIC FROM delta_ehr_cohorts
# MAGIC GROUP BY is_covid

# COMMAND ----------

# DBTITLE 1,Blood clots by age and gender
# MAGIC %sql
# MAGIC 
# MAGIC SELECT gender,
# MAGIC        age_at_blood_clot,
# MAGIC        count(*) as count
# MAGIC FROM delta_ehr_cohorts
# MAGIC WHERE is_blood_clot == 1
# MAGIC GROUP BY age_at_blood_clot, gender
# MAGIC ORDER BY age_at_blood_clot

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's compare admission rates among patinets who have received the treatment and those who have not

# COMMAND ----------

# DBTITLE 1,look at the admission probability for the target treatment
data_df.filter(f"is_{target_params['target_event_name']}==1")\
       .groupBy(f"is_{target_params['target_med_name']}")\
       .agg(
          avg(f"is_{target_params['target_outcome']}").alias('admission_probability'))\
       .display()

# COMMAND ----------

# MAGIC %md
# MAGIC as we see, overall the admission rates are lower among those who have received the traget treatmnet, however this can be confunded by many factors, for example it can be the case that younger patients are more likely receive the treatment and also being young less likely being admitted to the hospital. In this case we cannot attribute lower admission rates to the treatment. In the next notebook, we use propencity score matching to correct for such confunding factors. But first, let's write the resulting dataset to delta silver layer.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write final dataset to Delta
# MAGIC Now we write the resulting dataset back in the delta lake for our next analysis that is specifically looking into the effect of databrixovir on hospital admissions

# COMMAND ----------

data_df.write.mode('overwrite').save(f"{delta_path}/silver/patient_data")
