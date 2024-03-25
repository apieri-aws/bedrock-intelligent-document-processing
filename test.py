import boto3
import json

ssm = boto3.client('ssm')

def get_ssm_paramter(path):
    return ssm.get_parameter(Name=path, WithDecryption=True)
    
parameter_path = "/BedrockIDP/CLASSIFICATION"

response = get_ssm_paramter(parameter_path)

print(response)
print("")
print(response['Parameter']['Value'])
