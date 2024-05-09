from nltk.corpus import verbnet
from collections import defaultdict

def get_theta_roles(ex):
    theta_roles = set()
    for s in ex['semantics']:
        for a in s['arguments']:
            if a['type']=='ThemRole':
                theta_roles.add(a['value'])
    return theta_roles

def ditransitive_dobj(theta_roles):
    return ('Agent' in theta_roles and 'Destination' in theta_roles and 'Theme' in theta_roles) or \
            ('Agent' in theta_roles and 'Beneficiary' in theta_roles and 'Theme' in theta_roles)

def transitive_dobj(theta_roles):
    return 'Agent' in theta_roles and ('Theme' in theta_roles or 'Topic' in theta_roles)

def beneficiary(theta_roles):
    return 'Beneficiary' in theta_roles

def main():
    thematic_roles = defaultdict(set)
    thematic_roles_attr = defaultdict(set)
    for classid in verbnet.classids():
        f = verbnet.frames(classid)
        theta_roles = set()
        for ex in f:
            theta_roles.update(get_theta_roles(ex))
        for lemma in verbnet.lemmas(classid):
            thematic_roles[str(lemma)].update(theta_roles)
        thematic_roles[classid.split('-')[0]].update(theta_roles)
        
    with open('resources/ditransitive_dobj_verbs.txt', 'w') as outfile:
        for lemma in sorted(thematic_roles.keys()):
            if ditransitive_dobj(thematic_roles[lemma]) and '_' not in lemma:
                outfile.write(lemma+'\n')
                
    with open('resources/transitive_dobj_verbs.txt', 'w') as outfile:
        for lemma in sorted(thematic_roles.keys()):
            if transitive_dobj(thematic_roles[lemma]) and '_' not in lemma:
                outfile.write(lemma+'\n')
                
    with open('resources/benefactive_verbs.txt', 'w') as outfile:
        for lemma in sorted(thematic_roles.keys()):
            if beneficiary(thematic_roles[lemma]) and '_' not in lemma:
                outfile.write(lemma+'\n')
                
if __name__=='__main__':
    main()