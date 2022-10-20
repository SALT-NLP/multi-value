from .BaseDialect import BaseDialect
import pandas as pd
from collections import defaultdict

class DialectFromVector(BaseDialect):
    def __init__(self, vector=[], dialect_name=None, dialect_code=None, lexical_swaps={}, morphosyntax=True):
        super().__init__(lexical_swaps, morphosyntax)
        self.vector = vector
        self.dialect_name = dialect_name
        self.dialect_code = dialect_code
        self.surface_subs = [
            self.surface_lexical_sub,
            self.surface_contract,
            self.surface_fix_contracted_copula,
            self.surface_fix_spacing
        ]
        self.morphosyntax_transforms = {}
        self.feature_id_to_function_name = self.load_dict('resources/feature_id_to_function_name.json')
        self.attestation_vectors = pd.read_csv('resources/attestation_vectors.csv')
        
        if (type(self.dialect_name) == str) and (not len(self.vector)):
            if (self.dialect_name in self.attestation_vectors.columns):
                self.vector = self.attestation_vectors[self.dialect_name].values
            elif self.dialect_name == 'all':
                self.vector = [1.0]*len(self.attestation_vectors)
            
        self.initialize_from_vector()
        
    def initialize_from_vector(self):
        for idx, feature_attestation in enumerate(self.vector):
            feature_id = str(idx)
            if (feature_attestation) and (feature_id in self.feature_id_to_function_name):
                for feature_name in self.feature_id_to_function_name[feature_id]:
                    self.morphosyntax_transforms[feature_name] = feature_attestation
                    
class DialectFromFeatureList(BaseDialect):
    def __init__(self, feature_list=[], dialect_name=None, dialect_code=None, lexical_swaps={}, morphosyntax=True):
        super().__init__(lexical_swaps, morphosyntax)
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
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="all", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)        
        
class AboriginalDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Aboriginal English", dialect_code="AbE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class AfricanAmericanVernacular(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Urban African American Vernacular English", dialect_code="AAVE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class AppalachianDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Appalachian English", dialect_code="AppE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class AustralianDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Australian English", dialect_code="AusE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class AustralianVernacular(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Australian Vernacular English", dialect_code="AusVE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class BahamianDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Bahamian English", dialect_code="BahE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)

class BlackSouthAfricanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Black South African English", dialect_code="BlSAfE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class CameroonDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Cameroon English", dialect_code="CamE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class CapeFlatsDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Cape Flats English", dialect_code="CFE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class ChannelIslandsDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Channel Islands English", dialect_code="ChlIsE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class ChicanoDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Chicano English", dialect_code="ChcE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class ColloquialAmericanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Colloquial American English", dialect_code="CollAmE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class ColloquialSingaporeDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Colloquial Singapore English (Singlish)", dialect_code="CollSgE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class EarlyAfricanAmericanVernacular(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Earlier African American Vernacular English", dialect_code="EarlyAAVE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class EastAnglicanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="East Anglian English", dialect_code="EAngE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class FalklandIslandsDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Falkland Islands English", dialect_code="FlkIsE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class FijiAcrolect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Acrolectal Fiji English", dialect_code="FijAcE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class FijiBasilect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Pure Fiji English (basilectal FijiE)", dialect_code="FijE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class GhanaianDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Ghanaian English", dialect_code="GhE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class HongKongDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Hong Kong English", dialect_code="HKE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class IndianDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Indian English", dialect_code="IndE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class IndianSouthAfricanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Indian South African English", dialect_code="InSAfE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class IrishDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Irish English", dialect_code="IrE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class JamaicanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Jamaican English", dialect_code="JamE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class KenyanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Kenyan English", dialect_code="KenE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class LiberianSettlerDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Liberian Settler English", dialect_code="LibSE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class MalaysianDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Malaysian English", dialect_code="MalaE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class MalteseDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Maltese English", dialect_code="MaltE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class ManxDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Manx English", dialect_code="MxE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class NewZealandDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="New Zealand English", dialect_code="NZE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class NewfoundlandDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Newfoundland English", dialect_code="NfldE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class NigerianDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Nigerian English", dialect_code="NigE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class NorthEnglandDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="English dialects in the North of England", dialect_code="NEngE", lexical_swaps=lexical_swaps, morphosyntax=True)
        
class OrkneyShetlandDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Orkney and Shetland English", dialect_code="OrkShtE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class OzarkDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Ozark English", dialect_code="OzE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class PakistaniDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Pakistani English", dialect_code="PakE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class PhilippineDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Philippine English", dialect_code="PhilE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class RuralAfricanAmericanVernacular(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Rural African American Vernacular English", dialect_code="RuralAAVE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class ScottishDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Scottish English", dialect_code="ScE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class SoutheastAmericanEnglaveDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Southeast American enclave dialects", dialect_code="SeAmE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class SriLankanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Sri Lankan English", dialect_code="SrLaE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class StHelenaDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="St. Helena English", dialect_code="StHlE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class SoutheastEnglandDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="English dialects in the Southeast of England", dialect_code="SeEngE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class SouthwestEnglandDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="English dialects in the Southwest of England", dialect_code="SwEngE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class TanzanianDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Tanzanian English", dialect_code="TanE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class TristanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Tristan da Cunha English", dialect_code="TdCE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class UgandanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Ugandan English", dialect_code="UgE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class WelshDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="Welsh English", dialect_code="WelE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class WhiteSouthAfricanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="White South African English", dialect_code="WhSAfE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        
class WhiteZimbabweanDialect(DialectFromVector):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(dialect_name="White Zimbabwean English", dialect_code="WhZimE", lexical_swaps=lexical_swaps, morphosyntax=morphosyntax)
        

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
                         dialect_name="AAVE_Example_From_List")
        self.surface_subs = [
            self.surface_contract,
            self.surface_dey_conj,
            self.surface_aint_sub,
            self.surface_lexical_sub,
            self.surface_fix_contracted_copula,
            self.surface_fix_spacing
        ]