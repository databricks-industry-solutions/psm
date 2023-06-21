# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/psm. For more information about this solution accelerator, visit https://www.databricks.com/blog/2020/10/20/detecting-at-risk-patients-with-real-world-data.html.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Propensity score matching
# MAGIC Now that we have created the dataset needed for our analysis, in this notebook, we show how to perform propnecity score matching by training a classifier that assignes the probability of receiving a treatment based on patient's demographic information and past conditions. 
# MAGIC We then use this model to assign propencities to the target cohort and use stratififcation and nearest neighbor matching for PSM to estimate the effect of the treatment.

# COMMAND ----------

# MAGIC %pip install -U kaleido

# COMMAND ----------

import mlflow
from pyspark.sql.functions import *
import json
from pprint import pprint

# COMMAND ----------

# MAGIC %run ./config/00-config

# COMMAND ----------

# DBTITLE 1,configuration
with open(f'/tmp/{project_name}_configs.json','r') as f:
    settings = json.load(f)
    data_path = settings['data_path']
    base_path = settings['base_path']
    delta_path = settings['delta_path']
pprint(settings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create the propencity score model
# MAGIC Now, to estimate propencity scores, we use a `DecisionTreeClassifier` to classify patients into two classes of those who received the treatment and those who did not, based on selected demographic features as well as past history. We then apply the resulting model to the dataset to predict probabilities for belonging to each class which is the same as the propencity score for receiving the treatment. 

# COMMAND ----------

# DBTITLE 1,load data
data_df=spark.read.load(f"{delta_path}/silver/patient_data")

# COMMAND ----------

# MAGIC %md
# MAGIC Define target treatment, outcome of interest, and features to include in training for propencity score calculation

# COMMAND ----------

target_treatment='databrixovir'
target_cond='covid'
target_outcome='admission'

cathegorical_cols=['MARITAL',
 'RACE',
 'GENDER',
]

binary_cols=[
             'is_hypertension',
             'is_heart_disease',
]
info_col=['PATIENT']
label_column=f'is_{target_treatment}'
numerical_cols=[f'AGE_at_{target_cond}']
feature_columns=cathegorical_cols+binary_cols+numerical_cols

# COMMAND ----------

# DBTITLE 1,create training dataset
training_data_df=(
  data_df
  .filter(~isnull(label_column))
  .select(feature_columns+['PATIENT',label_column])
  .withColumnRenamed(label_column,'label')
)

# COMMAND ----------

# DBTITLE 1,distribution of labels
training_data_df.groupBy('label').count().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train the classifier
# MAGIC Now let's train the classifier. We use `DecisionTreeClassifier` from spark ml package. Note that you can also use any other binary classification method such as logistic regression or xgboost to assign propencity scores. See [this article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2907172/) for a review of different approaches for computing propencity scores.

# COMMAND ----------

# DBTITLE 1,train the model
from pyspark.ml.classification import DecisionTreeClassifier

from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler, VectorIndexer, Imputer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
import mlflow

mlflow.set_experiment(settings['experiment_name'])
params = {"split":0.8,
          "max_depth":5,
          "max_bins":32,
          "impurity":'entropy'
}
with mlflow.start_run(run_name='train-dtc') as run:
  input_cols = [f"{_col}_index" for _col in cathegorical_cols]+['age']+binary_cols
  train, test = training_data_df.randomSplit([params["split"], 1-params["split"]], seed=43)
  imputer = Imputer(inputCol='AGE_at_covid',outputCol='age')
  indexers = [StringIndexer(inputCol=_col, outputCol=f"{_col}_index", handleInvalid='keep') for _col in cathegorical_cols]
  assembler = VectorAssembler(inputCols=input_cols, outputCol="features",handleInvalid="keep")
  
  dt = DecisionTreeClassifier(labelCol="label", featuresCol="features",maxDepth=params['max_depth'],maxBins=params["max_bins"],impurity=params["impurity"])
  pipeline = Pipeline(stages=[imputer]+indexers+[assembler,dt])
  model = pipeline.fit(train)
  
  evaluator=BinaryClassificationEvaluator()
  accuracy=evaluator.evaluate(model.transform(test))
  print(f"the model accuracy is {accuracy:.2f}")
  
  mlflow.log_params(params)
  mlflow.log_metric('accuracy',accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC as we see in the above cell, the accuracy is `0.72` which indicates presence of bias in assigning treatment. 
# MAGIC Now let's take a closer look at the importance of different features, which can help us better understand which features 
# MAGIC are more relevant in deciding which patient is given the treatment. We also log the resulting plot as an mlflow artifact for future reference.
# MAGIC #👇🏻

# COMMAND ----------

# DBTITLE 1,feature importance
import pandas as pd
import plotly.express as px

input_features=[f"{_col}_index" for _col in cathegorical_cols]+['age']+binary_cols
fi_pdf=pd.DataFrame({'features':input_features,'importance':model.stages[-1].featureImportances.toArray()})

fig=px.bar(fi_pdf,x='features',y='importance')
mlflow.start_run(run_name='train-dtc')
fig.write_image('/dbfs/tmp/fi.png')
mlflow.log_artifact('/dbfs/tmp/fi.png')
mlflow.end_run()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explain paramters 
# MAGIC as we see above, treatment assignment is primarily driven by `AGE` followed by `GENDER` and `RACE`.
# MAGIC This is compatible with the ground truth data. As matter of fact we used these features to assign treatment in our simulated datasets. To further explore how these features determine the treatment assignment, we visualize the best tree in our random forest classifier by simply calling the `display` function on the classifier.
# MAGIC  #👇🏻

# COMMAND ----------

display(model.stages[-1])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate propencity scores
# MAGIC Now that we have a model to predict the propencity score for each patient, we can use this model to create a dataset of patients with their corresponding propencity scores.
# MAGIC To do so, we need to extract the probabilities from predictions of the model. This is done by defining a `udf` and applying it on the model output.

# COMMAND ----------

from pyspark.sql.types import DoubleType
get_prob=udf(lambda x:x.toArray().tolist()[1],DoubleType())

propencity_df=(
  model
  .transform(data_df.select(feature_columns+['PATIENT',f'is_{target_treatment}'])
  )
  .select('PATIENT','MARITAL','RACE','GENDER','is_hypertension','is_heart_disease','age',f'is_{target_treatment}',get_prob("probability").alias('p'))
)

display(propencity_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at propencity scores by age, which based on our previouse analysis is the most important feature in treatment assignment.

# COMMAND ----------

# DBTITLE 1,distribution p by age
import plotly.express as px

_pdf=propencity_df.toPandas()
_pdf['age_band']=10*(_pdf['age']//10)

fig = px.violin(_pdf, y="p",x="age_band",)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC the above plot indicates that older patients are more likely to receive the treatment,, however within each age group there is a mixed distribution of propencity scores, indicating dependence on other features (GENDER and RACE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Propencity Score Matching
# MAGIC Now that we have propencity scores, we proceed to creat propencity-matched pairs of patients from each case and control groups to assess the effect of the treatment under study.

# COMMAND ----------

# MAGIC %md
# MAGIC ### PSM Method 1: Stratification
# MAGIC There are two main approaches for propencity score matching. First we apply the stratification method:
# MAGIC > Stratification on the propensity score involves stratifying subjects into mutually exclusive subsets based on their estimated propensity score. Subjects are ranked according to their estimated propensity score.
# MAGIC
# MAGIC see [An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/) for more information.
# MAGIC We use [QuantileDiscretizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.QuantileDiscretizer.html#pyspark.ml.feature.QuantileDiscretizer.inputCols) to cretae different buckets for propencity scores

# COMMAND ----------

from pyspark.ml.feature import QuantileDiscretizer
qd=QuantileDiscretizer(inputCol="p", outputCol="p_strata",numBuckets=5)
qd_model=qd.fit(propencity_df)

# COMMAND ----------

# MAGIC %md
# MAGIC now we add _propencity bucket_ information for each patient

# COMMAND ----------

propencity_buckets_df=qd_model.transform(propencity_df).select('PATIENT','p_strata')

# COMMAND ----------

# MAGIC %md
# MAGIC next we calculate the fraction of admissions within each bucket

# COMMAND ----------

stratified_outcomes_df=data_df.select('PATIENT',f'is_{target_treatment}',f'is_{target_outcome}').join(propencity_buckets_df,on='PATIENT')
stratified_outcomes_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we calculate fraction of outcomes within each group

# COMMAND ----------

# DBTITLE 1,admission rates among different propensity strata
stratified_outcomes_df.groupBy(f'is_{target_treatment}','p_strata').agg(avg(f'is_{target_outcome}').alias('admission_rate')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC In the above plot we see that the admission rates among patients receiving the treatmnet is lower in all groups execpt the group with the lowest propencity to receive the treatment.

# COMMAND ----------

# MAGIC %md
# MAGIC ### PSM Method 1: Nearest Neighbor Method
# MAGIC Another method for estimnating the treatment effects and taking into account propencity scores, is to use the nearest neighbor method:
# MAGIC >Nearest neighbor matching selects for matching to a given treated subject that untreated subject whose propensity score is closest to that of the treated subject. If multiple untreated subjects have propensity scores that are equally close to that of the treated subject, one of these untreated subjects is selected at random.
# MAGIC
# MAGIC see [this review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/#R66) for more information on this method.
# MAGIC
# MAGIC Note that finding the nearest neighbor based on propencity scores requires pairwise comparsion of samples which can be computationaly expensive especially if we have a large sample size. Thankfully, when we leverage distributed computing with spark we can can easily find pairwise matches at scale.

# COMMAND ----------

# DBTITLE 1,choose case and control cohorts
cases=propencity_df.filter(f'is_{target_treatment}==1').selectExpr('PATIENT as P1','p as prop1').withColumn('rand',rand()).orderBy('rand').limit(5000)
controls=propencity_df.filter(f'is_{target_treatment}==0').selectExpr('PATIENT as P0','p as prop0')
n_cases=cases.count()
n_controls=controls.count()
print(f'number of cases:{n_cases}\nnumber of controls:{n_controls}')

# COMMAND ----------

# DBTITLE 1,create case-control pairs 
all_pairs=cases.join(controls,how='outer')
print(f'count of all pairs: {all_pairs.count()} = n_cases({n_cases})*n_controls({n_controls})')

# COMMAND ----------

all_pairs.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we select matched pairs based on the nearest neighbor method, i.e. for each case we choose the closest control based on propencity scores.
# MAGIC To do this we leverage window function

# COMMAND ----------

pairs=(
  all_pairs
  .repartition(1024)
  .selectExpr('*','abs(prop1-prop0) as p_diff')
  .selectExpr('P1','P0','prop1','prop0','row_number() over (PARTITION BY P1 order by p_diff) as rank')
  .filter('rank==1')
)

# COMMAND ----------

# MAGIC %md
# MAGIC since this is a computaionaly expensive task, it is good to write the results to disc for future reference:

# COMMAND ----------

pairs.write.mode('overWrite').option('overwriteSchema', 'true').save(f'{delta_path}/gold/nn-pairs')

# COMMAND ----------

# MAGIC %md
# MAGIC #### load stored data
# MAGIC Now let's read back the matched pairs and assign a unique id to each pair

# COMMAND ----------

pairs=spark.read.load(f'{delta_path}/gold/nn-pairs').selectExpr('P1','P0','uuid() as pair_id')
data_df=spark.read.load(f"{delta_path}/silver/patient_data")
target_treatment='databrixovir'
target_cond='covid'
target_outcome='admission'

# COMMAND ----------

pairs.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Now for each pair, we add outcome information (if admitted then the outcome is `1` and if not admitted then it is `0`).
# MAGIC The resuling dataframe is a dataframe where the first two columns are ids for cases and controls and the outcome for each pair is expressed as `00,01,10` and `11` corresponding to the outcome status of the pair (e.g `01` means that the case outcome is not admitted whereas the control's outcome is being admitted)

# COMMAND ----------

df1=(
  pairs
  .selectExpr('P1','pair_id','1 as treatment')
  .join(data_df.filter('is_covid==1').selectExpr('PATIENT as P1',f'is_{target_outcome}','int(is_admission) as O1'),on='P1')
)
df2=(
  pairs
  .selectExpr('P0','pair_id','0 as treatment')
  .join(data_df.filter('is_covid==1').selectExpr('PATIENT as P0',f'is_{target_outcome}','int(is_admission) as O0'),on='P0')
)
df=df1.join(df2,on='pair_id').selectExpr('P1','P0',"concat(O1,O0) as paired_outcome")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC For simplicity let's use the following notaions:
# MAGIC
# MAGIC - `a=n(11)` : number of pairs in which both the treated and the untreated subjects experience the event}
# MAGIC - `b=n(10)`: number of pairs in which the treated subject experiences the event, whereas the untreated subject does not
# MAGIC - `c=n(01)`: number of pairs in which the untreated subject experiences the event, whereas the treated subject does not
# MAGIC - `d=n(00)`: number of pairs in which both the treated and the untreated subjects do not experience the event
# MAGIC
# MAGIC The difference in the probability of the event between the treated and the untreated subjects is estimated by
# MAGIC
# MAGIC
# MAGIC \\[p_2-p_1 = (b-c)/n,\\]
# MAGIC
# MAGIC where `n` is the number of matched pairs and \\(p_2,p_1\\) are proportions of addmitted and not admitted patinets respectively.
# MAGIC
# MAGIC The variance of the difference in proportions is estimated by
# MAGIC
# MAGIC \\[v=\frac{(b+c)-(c-b)^2/n}{n^2} \\]
# MAGIC
# MAGIC see [Effects and non-effects of paired identical observations in comparing proportions with binary matched-pairs data](https://pubmed.ncbi.nlm.nih.gov/14695640/) for more details.

# COMMAND ----------

outcome_counts=df.groupBy('paired_outcome').count().toPandas()
outcome_counts

# COMMAND ----------

# MAGIC %md
# MAGIC then we have:
# MAGIC `a=n(11),b=n(10),c=n(01),d=n(00)`

# COMMAND ----------

a=int(outcome_counts[outcome_counts['paired_outcome']=='11']['count'])
b=int(outcome_counts[outcome_counts['paired_outcome']=='10']['count'])
c=int(outcome_counts[outcome_counts['paired_outcome']=='01']['count'])
d=int(outcome_counts[outcome_counts['paired_outcome']=='00']['count'])
n=df.count()

# COMMAND ----------

print(f"""difference in probability of the event is {(b-c)/n}
and the variance in the difference in proportions is: {((b+c)-(c-b)**2/n)/n**2}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC As we see in the avove equations, the estimtaed probability of hospital admissions for patinets who have received the target medication is lower than the control group (those who did not receive the medication), and the variance in the effect inidcates that the detected difference is statistically siginificant. 
