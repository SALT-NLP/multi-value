from .BaseDialect import BaseDialect
from collections import defaultdict
import geopy.distance
import numpy as np
import pandas as pd

class DialectFromVector(BaseDialect):
    def __init__(self, vector=[], dialect_name=None, **kwargs):
        super().__init__(**kwargs)
        self.vector = vector
        self.dialect_name = dialect_name
        self.surface_subs = [
            self.surface_lexical_sub,
            self.surface_contract,
            self.surface_fix_contracted_copula,
            self.surface_fix_spacing
        ]
        self.morphosyntax_transforms = {}
        self.feature_id_to_function_name = self.load_dict('resources/feature_id_to_function_name.json')
        self.attestation_vectors = pd.read_csv('resources/attestation_vectors.csv')
        
        self.dialect_code = None
        self.latitude = None
        self.longitude = None
        self.region = None
        
        varieties_ewave = pd.read_csv('resources/varieties.csv')
        consider = varieties_ewave[varieties_ewave['name']==dialect_name]
        if len(consider)==1:
            self.dialect_code = consider['abbr'].iloc[0]
            self.latitude = consider['latitude'].iloc[0]
            self.longitude = consider['longitude'].iloc[0]
            self.region = consider['region'].iloc[0]
        
        if (type(self.dialect_name) == str) and (not len(self.vector)):
            if (self.dialect_name in self.attestation_vectors.columns):
                self.vector = self.attestation_vectors[self.dialect_name].values
            elif self.dialect_name == 'all':
                self.vector = [1.0]*len(self.attestation_vectors)
            
        self.initialize_from_vector()

    def __str__(self):
        return f"""[VECTOR DIALECT] {self.dialect_name} (abbr: {self.dialect_code})
\tRegion: {self.region}
\tLatitude: {self.latitude}
\tLongitude: {self.longitude}"""
        
    def initialize_from_vector(self):
        for idx, feature_attestation in enumerate(self.vector):
            feature_id = str(idx)
            if (feature_attestation) and (feature_id in self.feature_id_to_function_name):
                for feature_name in self.feature_id_to_function_name[feature_id]:
                    self.morphosyntax_transforms[feature_name] = feature_attestation
                    
    def manhattan_distance(self, other, normalized=True):
        v1 = np.array(self.vector)
        v2 = np.array(other.vector)
        diff = np.abs(v1-v2)
        if normalized:
            return diff.sum() / len(diff)
        return diff.sum()
    
    def geographical_distance(self, other, metric=True):
        if metric:
            return geopy.distance.geodesic( (self.latitude, self.longitude), (other.latitude, other.longitude)).km
        return geopy.distance.geodesic( (self.latitude, self.longitude), (other.latitude, other.longitude)).miles
                    
class DialectFromFeatureList(BaseDialect):
    def __init__(self, feature_list=[], dialect_name=None, dialect_code=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_list = feature_list
        self.dialect_name = dialect_name
        self.surface_subs = [
            self.surface_lexical_sub,
            self.surface_contract,
            self.surface_fix_contracted_copula,
            self.surface_fix_spacing
        ]
        self.morphosyntax_transforms = {}
        
        self.initialize_from_feature_list()
        
    def initialize_from_feature_list(self):
        for feature_name in self.feature_list:
            self.morphosyntax_transforms[feature_name] = 1.0
        
class MultiDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="all", dialect_code="MULTI", **kwargs)
        
class AboriginalDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Aboriginal English", **kwargs)
        
class AfricanAmericanVernacular(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Urban African American Vernacular English", **kwargs)
        
class AppalachianDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Appalachian English", **kwargs)
        
class AustralianDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Australian English", **kwargs)
        
class AustralianVernacular(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Australian Vernacular English", **kwargs)
        
class BahamianDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Bahamian English", **kwargs)

class BlackSouthAfricanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Black South African English", **kwargs)
        
class CameroonDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Cameroon English", **kwargs)
        
class CapeFlatsDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Cape Flats English", **kwargs)
        
class ChannelIslandsDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Channel Islands English", **kwargs)
        
class ChicanoDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Chicano English", **kwargs)
        
class ColloquialAmericanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Colloquial American English", **kwargs)
        
class ColloquialSingaporeDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Colloquial Singapore English (Singlish)", **kwargs)
        
class EarlyAfricanAmericanVernacular(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Earlier African American Vernacular English", **kwargs)
        
class EastAnglicanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="East Anglian English", **kwargs)
        
class FalklandIslandsDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Falkland Islands English", **kwargs)
        
class FijiAcrolect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Acrolectal Fiji English", **kwargs)
        
class FijiBasilect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Pure Fiji English (basilectal FijiE)", **kwargs)
        
class GhanaianDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Ghanaian English", **kwargs)
        
class HongKongDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Hong Kong English", **kwargs)
        
class IndianDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Indian English", **kwargs)
        
class IndianSouthAfricanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Indian South African English", **kwargs)
        
class IrishDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Irish English", **kwargs)
        
class JamaicanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Jamaican English", **kwargs)
        
class KenyanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Kenyan English", **kwargs)
        
class LiberianSettlerDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Liberian Settler English", **kwargs)
        
class MalaysianDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Malaysian English", **kwargs)
        
class MalteseDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Maltese English", **kwargs)
        
class ManxDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Manx English", **kwargs)
        
class NewZealandDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="New Zealand English", **kwargs)
        
class NewfoundlandDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Newfoundland English", **kwargs)
        
class NigerianDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Nigerian English", **kwargs)
        
class NorthEnglandDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="English dialects in the North of England", **kwargs)
        
class OrkneyShetlandDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Orkney and Shetland English", **kwargs)
        
class OzarkDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Ozark English", **kwargs)
        
class PakistaniDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Pakistani English", **kwargs)
        
class PhilippineDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Philippine English", **kwargs)
        
class RuralAfricanAmericanVernacular(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Rural African American Vernacular English", **kwargs)
        
class ScottishDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Scottish English", **kwargs)
        
class SoutheastAmericanEnglaveDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Southeast American enclave dialects", **kwargs)
        
class SriLankanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Sri Lankan English", **kwargs)
        
class StHelenaDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="St. Helena English", **kwargs)
        
class SoutheastEnglandDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="English dialects in the Southeast of England", **kwargs)
        
class SouthwestEnglandDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="English dialects in the Southwest of England", **kwargs)
        
class TanzanianDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Tanzanian English", **kwargs)
        
class TristanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Tristan da Cunha English", **kwargs)
        
class UgandanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Ugandan English", **kwargs)
        
class WelshDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="Welsh English", **kwargs)
        
class WhiteSouthAfricanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="White South African English", **kwargs)
        
class WhiteZimbabweanDialect(DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="White Zimbabwean English", **kwargs)
        

# Other examples
        
class AAVE_Example_From_List(DialectFromFeatureList):
    def __init__(self):
        super().__init__(feature_list=['adj_for_adv',
                                         'analytic_or_double_superlative',
                                         'ass_pronoun',
                                         'completive_been_done',
                                         'demonstrative_for_definite_articles',
                                         'double_comparative',
                                         'double_modals',
                                         'drop_aux',
                                         'existential_dey_it',
                                         'got',
                                         'irrealis_be_done',
                                         'mass_noun_plurals',
                                         'negative_concord',
                                         'negative_inversion',
                                         'null_genitive',
                                         'null_relcl',
                                         'participle_or_bare_past_tense',
                                         'past_for_past_participle',
                                         'plural_interrogative',
                                         'referential_thing',
                                         'reflex_number',
                                         'regularized_plurals',
                                         'regularized_reflexives',
                                         'regularized_reflexives_aave',
                                         'shadow_pronouns',
                                         'synthetic_comparative',
                                         'that_what_relativizer',
                                         'uninflect',
                                         'what_comparative',
                                         'will_would'],
                         dialect_name="AAVE_Example_From_List",
                         dialect_code="AAVE_Example"
                        )
        self.surface_subs = [
            self.surface_contract,
            self.surface_dey_conj,
            self.surface_aint_sub,
            self.surface_lexical_sub,
            self.surface_fix_contracted_copula,
            self.surface_fix_spacing
        ]