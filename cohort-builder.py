# Databricks notebook source
# MAGIC %md
# MAGIC This class contains methods for:
# MAGIC   1. Loading synthea resources
# MAGIC   2. Creating cohorts correspodning to a given phenotype
# MAGIC   3. Creating datasets with binary cohort inclusion/exclusion indicator variables
# MAGIC 
# MAGIC methods:
# MAGIC ```
# MAGIC set_patient_list(demog_list=None)
# MAGIC add_cohort(resource, event_name, event_codes,hist_window,washout_window, target_event_name):
# MAGIC add_cohort(resource, event_name, event_codes,hist_window,washout_window, target_event_name)
# MAGIC get_cohort_tags()
# MAGIC ```
# MAGIC 
# MAGIC example:
# MAGIC ```
# MAGIC my_ehr=DeltaEHR(<path/to/ehr_records>)
# MAGIC my_ehr.add_simple_cohort(cohort_name='covid',resource='conditions',inclusion_criteria='CODE==840539006')
# MAGIC my_ehr.set_patient_list(demog_list=['BIRTHDATE','MARITAL','RACE','ETHNICITY','GENDER','HEALTHCARE_COVERAGE'])
# MAGIC ```

# COMMAND ----------

import re
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, DoubleType
import mlflow
import json

# COMMAND ----------

class DeltaEHR:
  """
  this class is designed to ingest synthetic patient records and build cohorts based on user-defined inclusion, exclusion criteria,
  outcomes of interest and comorbidities.
  """
  
  def __init__(self,delta_path):
    
    self.delta_path=delta_path
    self.tables = {}
    for resource in [m.name.strip('/') for m in dbutils.fs.ls(f'{delta_path}/bronze/')]:
      self.tables[resource] = spark.read.load(f'{delta_path}/bronze/{resource}')
    
    self.tables['patients']=self.tables['patients'].withColumnRenamed('Id','PATIENT')
    self.cohorts = {}
    self.patient_list = None
  
  def set_patient_list(self,demog_list=None):
    """
    Super list of all patinets to inlclude plus start, stop times for records.
    This can be 
    """
    if demog_list==None:
      demog_list=['BIRTHDATE','MARITAL','RACE','ETHNICITY','GENDER','BIRTHPLACE','CITY','STATE','COUNTY','ZIP']
    
    self.tables['encounters'].createOrReplaceTempView('encounters')

    patients_with_enc_dates = sql("""
      select PATIENT,max(to_date(START)) as max_START, min(to_date(STOP)) as min_STOP
      from encounters
      group By PATIENT
    """
    )
    
    self.patient_list = self.tables['patients'].select(['PATIENT']+demog_list).join(patients_with_enc_dates,on="PATIENT")
    
  
  def add_simple_cohort(self,cohort_name,resource,inclusion_criteria,start_col='START',stop_col='STOP'):
    """
    adding cohorts based on inclusion criteria expressed as a sql expression.
    """
    
    assert start_col in self.tables[resource].columns
    assert stop_col in self.tables[resource].columns

    cohort_df =  (self.tables[resource]
                 .filter(inclusion_criteria)
                 .selectExpr('PATIENT',f'to_date({start_col}) as START',f'to_date({stop_col}) as STOP')
                 .dropDuplicates()
                )
    
    self.cohorts[cohort_name]=cohort_df
  
  def add_cohort(self, resource, event_name, event_codes, hist_window, washout_window, target_event_name):
    """
    add cohorts based on prior events (comorbidities, drug exposure etc)
    """
    assert 'START' in self.tables[resource].columns
    assert 'STOP'  in  self.tables[resource].columns
    
    target_cohort=self.cohorts[target_event_name]
    
    sql_expr=f"""  
      datediff({target_event_name}_START,{event_name}_START)<{hist_window}
      AND datediff({target_event_name}_START,{event_name}_START)>{washout_window}
    """
    event_df=(
      self.tables[resource]
      .filter(col('CODE').isin(event_codes))
      .selectExpr('PATIENT',f'START as {event_name}_START',f'STOP as {event_name}_STOP')
      .join(target_cohort.selectExpr('PATIENT',f'START as {target_event_name}_START',f'STOP as {target_event_name}_STOP'),on="PATIENT")
      .filter(sql_expr)
      .selectExpr("PATIENT",f'{event_name}_START as START',f'{event_name}_STOP as STOP')
      .selectExpr('PATIENT','START','STOP','rank() over (partition by PATIENT ORDER BY START DESC) AS rank')
      .filter('rank==1')
    )
    
    self.cohorts[event_name]=event_df.select('PATIENT',f'START',f'STOP')
  
  def combine_all_cohorts(self):
    """
    combine all cohort information as columns of indicator functions (cohort membership indicator) and temporal information for each cohort
    """
    if self.patient_list==None:
      self.set_patient_list()
    
    cohort_tags=self.patient_list
    
    for c_name,c_df in self.cohorts.items():
      html_str=f"adding cohort {c_name} ..."
      displayHTML(html_str)
      cohort_tags=(
        cohort_tags
        .join(c_df.selectExpr('PATIENT',f'START AS {c_name}_START'),on='PATIENT',how='left')
        .withColumn(f"is_{c_name}",(~isnull(f'{c_name}_START')).cast(DoubleType()))
        .withColumn(f"age_at_{c_name}",year(f'{c_name}_START')-year(to_date('BIRTHDATE')))
      )
    return(cohort_tags)
