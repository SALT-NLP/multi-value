import unittest
from value import BaseDialect, Dialects
import pandas as pd
import json

class TestStringMethods(unittest.TestCase):
    
    def test_all_methods(self):
        D = Dialects.DialectFromVector(dialect_name="all")
        feature_id_to_function_name = D.load_dict('resources/feature_id_to_function_name.json')
        uts = D.load_dict('resources/unittests.json')
        
        failure_cases = 0
        for feature_id in uts:
            for function_name in feature_id_to_function_name[feature_id]:
                for ut in uts[feature_id]:
                    sae = ut['gloss']
                    dialect = ut['dialect']
                    D.clear()
                    D.update(sae)
                    method = getattr(D, function_name)
                    method()
                    synth_dialect = D.surface_fix_spacing(D.compile_from_rules())
                    if synth_dialect.lower() not in {x.lower() for x in dialect}:
                        failure_cases += 1
                        print(feature_id, function_name, dialect, synth_dialect)
        assert(failure_cases==0)

