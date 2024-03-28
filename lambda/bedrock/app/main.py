"""
kicks off Step Function executions
"""
import json
import logging
import os
import textractmanifest as tm
import boto3
from uuid import uuid4

from typing import Tuple

logger = logging.getLogger(__name__)

version = "0.0.1"
s3 = boto3.client("s3")
ssm = boto3.client('ssm')
bedrock_rt = boto3.client("bedrock-runtime",region_name="us-east-1")


def split_s3_path_to_bucket_and_key(s3_path: str) -> Tuple[str, str]:
    if len(s3_path) > 7 and s3_path.lower().startswith("s3://"):
        s3_bucket, s3_key = s3_path.replace("s3://", "").split("/", 1)
        return (s3_bucket, s3_key)
    else:
        raise ValueError(
            f"s3_path: {s3_path} is no s3_path in the form of s3://bucket/key."
        )


def get_file_from_s3(s3_path: str, range=None) -> bytes:
    s3_bucket, s3_key = split_s3_path_to_bucket_and_key(s3_path)
    if range:
        o = s3.get_object(Bucket=s3_bucket, Key=s3_key, Range=range)
    else:
        o = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    return o.get("Body").read()


class ThrottlingException(Exception):
    pass


class ServiceQuotaExceededException(Exception):
    pass


class ModelTimeoutException(Exception):
    pass


class ModelNotReadyException(Exception):
    pass


def get_ssm_paramter(path):
    return ssm.get_parameter(Name=path, WithDecryption=True)

def generate_message(bedrock_runtime, model_id, system, messages, max_tokens,top_p, temp):
    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "temperature": temp,
            "top_p": top_p,
        }  
    )  
    
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())

    return response_body


def lambda_handler(event, _):
    """
    if the FIXED_KEY is empty, it takes the classification result to find the prompt.
    if the FIXED_KEY is set, it will always execute that prompt, which is useful for classification
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    logger.setLevel(log_level)
    logger.info(f"LOG_LEVEL: {log_level}")
    logger.info(json.dumps(event))
    bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
    fixed_key = os.environ.get("FIXED_KEY", None)
    s3_output_bucket = os.environ.get('S3_OUTPUT_BUCKET')
    s3_output_prefix = os.environ.get('S3_OUTPUT_PREFIX')
    
    if not s3_output_bucket:
        raise Exception("no S3_OUTPUT_BUCKET set")
    if not s3_output_prefix:
        raise Exception("no S3_OUTPUT_PREFIX set")
    # Get manifest file from event context
    try:
        if "Payload" in event and "manifest" in event["Payload"]:
            manifest: tm.IDPManifest = tm.IDPManifestSchema().load(event["Payload"]["manifest"])  # type: ignore
        elif "manifest" in event:
            manifest: tm.IDPManifest = tm.IDPManifestSchema().load(event["manifest"])  # type: ignore else:
        else:
            manifest: tm.IDPManifest = tm.IDPManifestSchema().load(event)  # type: ignore

        # fixed_key for classification and non fixed key for extraction
        if fixed_key:
            parameter_name = fixed_key
        else:
            if "classification" in event and "documentType" in event["classification"]:
                parameter_name = event["classification"]["documentType"]
                logger.debug(f"document_type: {parameter_name}")
            else:
                raise ValueError(
                    f"no [classification][documentType] given in event: {event}"
                )
                
        # Get SSM parameter path for prompt        
        parameter_path = f"/BedrockIDP/{parameter_name}"
        ssm_response = get_ssm_paramter(parameter_path)

        # Load prompt from SSM
        prompt = ""
        
        if (
            "Parameter" in ssm_response
            and "Value" in ssm_response["Parameter"]
        ):
            prompt = ssm_response["Parameter"]["Value"]
        else:
            raise ValueError(
                "no ['Value'] in parameter store "
            )
            
        # Load text document from S3
        if (
            "txt_output_location" in event
            and "TextractOutputCSVPath" in event["txt_output_location"]
        ):
            document_text_path = event["txt_output_location"]["TextractOutputCSVPath"]
        else:
            raise ValueError(
                "no ['txt_output_location']['TextractOutputCSVPath'] to get the text file from "
            )

        document_text = get_file_from_s3(s3_path=document_text_path).decode('utf-8')
        
        system_config = f"You are an AI assistant that performs document classification and extraction tasks. Given the following document <text>{{{document_text}}}</text>, answer the user's questions"
        
        logger.debug(system_config)
        
        message_config = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
                
        response = generate_message(
            bedrock_runtime=bedrock_rt, model_id=bedrock_model_id, system=system_config, messages=message_config, max_tokens=512, temp=0.2, top_p=0.9
        )
        
        if "completion" in response:
            output_text = response['completion']
        elif "content" in response:
            output_text =response['content'][0]['text']
            logger.debug(output_text)
        else:
            output_text = ""

        logger.debug(response)

        # If our task is classification, add the document classification to the event context
        if parameter_name == "CLASSIFICATION":
            classification_json = json.loads(output_text)
            document_type = classification_json['CLASSIFICATION']
            event.setdefault('classification', {})['documentType'] = document_type
        
        # Write the document classification to S3
        s3_filename, _ = os.path.splitext(os.path.basename(manifest.s3_path))
        output_bucket_key = s3_output_prefix + "/" + s3_filename + str(uuid4()) + ".json"
        s3.put_object(Body=bytes(
            output_text.encode('UTF-8')),
                    Bucket=s3_output_bucket,
                    Key=output_bucket_key)

    except bedrock_rt.exceptions.ThrottlingException as te:
        raise ThrottlingException(te)

    except bedrock_rt.exceptions.ModelNotReadyException as mnr:
        raise ModelNotReadyException(mnr)

    except bedrock_rt.exceptions.ModelTimeoutException as mt:
        raise ModelTimeoutException(mt)

    except bedrock_rt.exceptions.ServiceQuotaExceededException as sqe:
        raise ServiceQuotaExceededException(sqe)

    event['bedrock_output'] = f"s3://{s3_output_bucket}/{output_bucket_key}"
    return event