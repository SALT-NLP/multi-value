import boto3
import argparse
import os
from collections import defaultdict

region_name = 'us-east-1'
aws_access_key_id = ''
aws_secret_access_key = ''

#endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

# Uncomment this line to use in production
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

def get_workers_with_qual(mturk, qual_id):
    worker_ids = set()

    nextToken = None
    response = {}
    while True:
        if nextToken:
            response = mturk.list_workers_with_qualification_type(
                QualificationTypeId=qual_id,
                Status='Granted',
                NextToken=nextToken,
                MaxResults=100
            )
        else:
            response = mturk.list_workers_with_qualification_type(
                QualificationTypeId=qual_id,
                Status='Granted',
                MaxResults=100
            )

        for q in response['Qualifications']:
            worker_ids.add(q['WorkerId'])

        if 'NextToken' not in response:
            break
        nextToken = response['NextToken']

    return worker_ids

def main():

    dialect_codes = ['aboriginal', 'fiji_acrolectal', 'appalachian', 'australian',
    'australian_vernacular', 'bahamian', 'south_african_black', 'cameroon', 'cape_flats',
    'channel_islands', 'chicano', 'colloquial_ae', 'singapore', 'aave_earlier',
    'east_anglian', 'england_n', 'england_se', 'england_sw', 'falkland_islands',
    'ghanaian', 'hk', 'indian', 'indian_south_african', 'irish', 'jamaican', 'kenyan',
    'liberian', 'malaysian', 'maltese', 'manx', 'new_zealand', 'newfoundland',
    'nigerian', 'orkney', 'ozark', 'pakistani', 'philippine', 'fiji_pure', 'aave_rural',
    'scottish', 'se_american_enclave', 'sri_lankan', 'st_helena', 'tanzanian',
    'tristan_da_cunha', 'ugandan', 'aave_urban', 'welsh', 'south_african_white', 'zimbabwean_white', 'other']

    dialect_names = defaultdict(lambda: "OTHER")
    dialect_names['aave_urban'] = 'A'
    dialect_names['aave_rural'] = 'A'
    dialect_names['aave_earlier'] = 'A'
    dialect_names['singapore'] = 'B'
    dialect_names['indian'] = 'C'
    dialect_names['appalachian'] = 'D'
    dialect_names['chicano'] = 'E'
    dialect_names['colloquial_ae'] = 'F'
    dialect_names['aboriginal'] = 'G'
    dialect_names['se_american_enclave'] = 'H'
    dialect_names['england_n'] = 'I'
    dialect_names['south_african_black'] = 'J'
    dialect_names['indian_south_african'] = 'K'
    dialect_names['east_anglian'] = 'L'
    dialect_names['malaysian'] = 'M'
    dialect_names['south_african_white'] = 'N'
    dialect_names['ozark'] = 'O'
    dialect_names['philippine'] = 'P'

    dialect_qual_mapping = {
        'A': args.A, # dialect A 
        'B': args.B, # dialect B 
        'C': args.C, # dialect C 
        'D': args.D, # dialect D 
        'E': args.E, # dialect E
        'F': args.F, # dialect F
        'G': args.G, # dialect G
        'H': args.H, # dialect H
        'I': args.I, # dialect I
        'J': args.J, # dialect J
        'K': args.K, # dialect K
        'L': args.L, # dialect L
        'M': args.M, # dialect M
        'N': args.N, # dialect N
        'O': args.O, # dialect O
        'P': args.P, # dialect P
        'Q': args.Q, # dialect Q
        'R': args.R, # dialect R
    }


    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    workers_with_quals = {
        'A': get_workers_with_qual(mturk, dialect_qual_mapping['A']),
        'B': get_workers_with_qual(mturk, dialect_qual_mapping['B']),
        'C': get_workers_with_qual(mturk, dialect_qual_mapping['C']),
        'D': get_workers_with_qual(mturk, dialect_qual_mapping['D']),
        'E': get_workers_with_qual(mturk, dialect_qual_mapping['E']),
        'F': get_workers_with_qual(mturk, dialect_qual_mapping['F']),
        'G': get_workers_with_qual(mturk, dialect_qual_mapping['G']),
        'H': get_workers_with_qual(mturk, dialect_qual_mapping['H']),
        'I': get_workers_with_qual(mturk, dialect_qual_mapping['I']),
        'J': get_workers_with_qual(mturk, dialect_qual_mapping['J']),
        'K': get_workers_with_qual(mturk, dialect_qual_mapping['K']),
        'L': get_workers_with_qual(mturk, dialect_qual_mapping['L']),
        'M': get_workers_with_qual(mturk, dialect_qual_mapping['M']),
        'N': get_workers_with_qual(mturk, dialect_qual_mapping['N']),
        'O': get_workers_with_qual(mturk, dialect_qual_mapping['O']),
        'P': get_workers_with_qual(mturk, dialect_qual_mapping['P']),
        'Q': get_workers_with_qual(mturk, dialect_qual_mapping['Q']),
        'R': get_workers_with_qual(mturk, dialect_qual_mapping['R'])
    }


    nextToken = None
    response = {}

    while True:
        if nextToken:
            response = mturk.list_workers_with_qualification_type(
                QualificationTypeId=args.qual_id,
                Status='Granted',
                NextToken=nextToken,
                MaxResults=100
            )
        else:
            response = mturk.list_workers_with_qualification_type(
                QualificationTypeId=args.qual_id,
                Status='Granted',
                MaxResults=100
            )

        for q in response['Qualifications']:
            workerId = q['WorkerId']
            code = str(q['IntegerValue'])
            try:
                self_report_code = dialect_codes[int(code[-2:])-1]
                quiz_result_code = dialect_codes[int(code[:-2])-1]
            except:
                print('error with', code)
                continue

            print(workerId, self_report_code, quiz_result_code)

            if ((self_report_code in dialect_names) and (quiz_result_code in dialect_names)):
                self_report = dialect_names[self_report_code]
                quiz_result = dialect_names[quiz_result_code]

                if True: # (self_report == quiz_result): # might have to adjust this later...
                    if not workerId in workers_with_quals[self_report]:
                        mturk.associate_qualification_with_worker(
                            QualificationTypeId=dialect_qual_mapping[self_report],
                            WorkerId=workerId,
                            IntegerValue=1,
                            SendNotification=True
                        )
                        print('Assigned', workerId, self_report, '(', self_report_code, ')')

        if 'NextToken' not in response:
            break
        nextToken = response['NextToken']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qual_id', type=str, help='the name of the QualificationTypeId for the Dialect Qual')
    parser.add_argument('--A', type=str, help='the name of the QualificationTypeId for Dialect A')
    parser.add_argument('--B', type=str, help='the name of the QualificationTypeId for Dialect B')
    parser.add_argument('--C', type=str, help='the name of the QualificationTypeId for Dialect C')
    parser.add_argument('--D', type=str, help='the name of the QualificationTypeId for Dialect D')
    parser.add_argument('--E', type=str, help='the name of the QualificationTypeId for Dialect E')
    parser.add_argument('--F', type=str, help='the name of the QualificationTypeId for Dialect F')
    parser.add_argument('--G', type=str, help='the name of the QualificationTypeId for Dialect G')
    parser.add_argument('--H', type=str, help='the name of the QualificationTypeId for Dialect H')
    parser.add_argument('--I', type=str, help='the name of the QualificationTypeId for Dialect I')
    parser.add_argument('--J', type=str, help='the name of the QualificationTypeId for Dialect J')
    parser.add_argument('--K', type=str, help='the name of the QualificationTypeId for Dialect K')
    parser.add_argument('--L', type=str, help='the name of the QualificationTypeId for Dialect L')
    parser.add_argument('--M', type=str, help='the name of the QualificationTypeId for Dialect M')
    parser.add_argument('--N', type=str, help='the name of the QualificationTypeId for Dialect N')
    parser.add_argument('--O', type=str, help='the name of the QualificationTypeId for Dialect O')
    parser.add_argument('--P', type=str, help='the name of the QualificationTypeId for Dialect P')
    parser.add_argument('--Q', type=str, help='the name of the QualificationTypeId for Dialect Q')
    parser.add_argument('--R', type=str, help='the name of the QualificationTypeId for Dialect R')
    args = parser.parse_args()
    main()
