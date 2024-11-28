import json
import os
import re
import random
from math import factorial
from itertools import combinations, product, islice
from spacy.lang.en import English
from manual_inference.helper import get_schema_sdss
from intermediate_representation.semQL import Root1, Root, Sel, Sup, N, A, C, T, Filter, V, Op
from data_augmentation.sql_generator import sql2semQL
from intermediate_representation.sem2sql.sem2SQL import transform


dataset = [
  {
    "db_id": "all_trial_metadata",
    "query": "SELECT * FROM cdms_trl; ",
    "question": "List all clinical trials."
  },
  {
    "db_id": "all_trial_metadata",
    "query": "SELECT trl_id FROM trl_type WHERE type_of_trl = ‘SAFETY’",
    "question": "List all clinical trial protocols where the trial is a safety trial."
  },
  {
    "db_id": "all_trial_metadata",
    "query": "SELECT trl_id FROM plnd_comp_in_trl WHERE topic_cd = 'INSULIN DETEMIR' AND (device = 'PREFILLED PEN' OR device = 'FLEXPEN')",
    "question": "Find all clinical trial protocols that target “insulin” where the device used is either a prefilled pen or the flexpen."
  },
  {
    "db_id": "all_trial_metadata",
    "query": "SELECT trl_type.trl_id, trl_type.type_of_trl, trl_site.site_id FROM trl_type INNER JOIN trl_site ON trl_type.trl_id = trl_site.trl_id where trl_type.type_of_trl = 'EFFICACY' OR trl_type.type_of_trl='SAFETY'",
    "question": "List all clinical safety and efficacy trial protocols along with the place of the trial."
  },
  {
    "db_id": "all_trial_metadata",
    "query": "SELECT type_of_trl, COUNT(type_of_trl) FROM trl_type GROUP BY type_of_trl ORDER BY COUNT(type_of_trl) DESC",
    "question": "Show an overview of the different trial types and the amount conducted of each in descending order."
  },
  {
    "db_id": "all_trial_metadata",
    "query": "SELECT DISTINCT trl_id FROM plnd_comp_dos_in_elem WHERE trl_id IN (SELECT trl_id FROM plnd_comp_in_trl WHERE comp_strength = '100 U/mL' AND device = 'FLEXPEN' ) AND dose_freq_unit = 'WEEKLY'",
    "question": "Find trials that are conducted using a weekly frequency for doses, with a dose size of 100 U/ml using the flexpen."
  },
  {
    "db_id": "all_trial_metadata",
    "query": "SELECT COUNT(DISTINCT plnd_comp_in_trl.trl_id) FROM plnd_comp_in_trl INNER JOIN plnd_trl_elem ON plnd_comp_in_trl.trl_id = plnd_trl_elem.trl_id WHERE plnd_comp_in_trl.device = 'FLEXPEN' AND ((plnd_trl_elem.dur >= 28 AND plnd_trl_elem.dur_unit = 'DAY') OR (plnd_trl_elem.dur >= 4 AND plnd_trl_elem.dur_unit = 'WEEK'))",
    "question": "Find the number of trials conducted using the FLEXPEN for more than 28 days."
  }
]

dir_path = os.path.dirname(__file__)
par_path = os.path.dirname(dir_path)
root_path = os.path.dirname(par_path)
data_path = os.path.join(root_path,'data')
schema_dict_path = "TrialBench/original/tables.json"
schema_dict_path = os.path.join(data_path, schema_dict_path)
with open(schema_dict_path, 'r') as input:
    _schema_dict = json.load(input)[0]
schema_dict = {_schema_dict['db_id']: _schema_dict}


def main():

    data, table = sql2semQL(dataset=dataset, schema_dict=schema_dict, table_file=schema_dict_path)
    print(data[0]['sql']['select'])
    print()
    print(data[0]['sql']['where'])
    print()
    print(data[0]['sql']['orderBy'])
    print()
    print(data[0]['rule_label'])
    res = []
    
    #for d in data:
    #    res.append(
    #        transform(d, table['skyserver_dr16_2020_11_30'], origin=d['rule_label']))
    #print(*res, sep='\n')
    


if __name__ == '__main__':
    main()


"""
Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(8) T(1) A(0) Op(0) C(8) T(4) Filter(0) Filter(5) A(0) Op(0) C(75) T(1) V(0) Filter(0) Filter(1) Filter(4) A(0) Op(0) C(75) T(1) V(1) Filter(4) A(0) Op(0) C(80) T(1) V(2) Filter(0) Filter(2) A(0) Op(0) C(15) T(1) V(3) Filter(0) Filter(6) A(0) Op(0) C(124) T(4) V(4) Filter(2) A(0) Op(0) C(127) T(4) V(5)

values: ['0', '23', '0.2', '1', '0.1', 'GALAXY']
"""