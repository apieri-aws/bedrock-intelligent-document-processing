"""
kicks off Step Function executions
"""
import json
import logging
import os
import textractmanifest as tm
import boto3
import jinja2
import base64
from uuid import uuid4

from typing import Tuple
from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute
from pynamodb.exceptions import DoesNotExist

logger = logging.getLogger(__name__)

version = "0.0.1"
s3 = boto3.client("s3")
bedrock_rt = boto3.client("bedrock-runtime",region_name="us-east-1")


def split_s3_path_to_bucket_and_key(s3_path: str) -> Tuple[str, str]:
    if len(s3_path) > 7 and s3_path.lower().startswith("s3://"):
        s3_bucket, s3_key = s3_path.replace("s3://", "").split("/", 1)
        return (s3_bucket, s3_key)
    else:
        raise ValueError(
            f"s3_path: {s3_path} is no s3_path in the form of s3://bucket/key."
        )


def get_image_file_from_s3(s3_path: str, range=None) -> bytes:
    s3_bucket, s3_key = split_s3_path_to_bucket_and_key(s3_path)
    if range:
        o = s3.get_object(Bucket=s3_bucket, Key=s3_key, Range=range)
    else:
        o = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    return base64.b64encode(o.get("Body").read())


class ThrottlingException(Exception):
    pass


class ServiceQuotaExceededException(Exception):
    pass


class ModelTimeoutException(Exception):
    pass


class ModelNotReadyException(Exception):
    pass


class BedrockPromptConfig(Model):
    class Meta:
        table_name = os.environ["BEDROCK_CONFIGURATION_TABLE"]
        region = boto3.Session().region_name

    id = UnicodeAttribute(hash_key=True, attr_name="id")
    prompt_template = UnicodeAttribute(attr_name="v")

def generate_message(bedrock_runtime, model_id, messages, max_tokens,top_p, temp):

    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temp,
            "top_p": top_p
        }  
    )  
    
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())

    return response_body


def lambda_handler(event, _):
    """
    Reads a jinja2 template from the DDB table passed in as BEDROCK_CONFIGURATION_TABLE
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

    try:
        if "Payload" in event and "manifest" in event["Payload"]:
            manifest: tm.IDPManifest = tm.IDPManifestSchema().load(event["Payload"]["manifest"])  # type: ignore
        elif "manifest" in event:
            manifest: tm.IDPManifest = tm.IDPManifestSchema().load(event["manifest"])  # type: ignore else:
        else:
            manifest: tm.IDPManifest = tm.IDPManifestSchema().load(event)  # type: ignore

        # fixed_key for classification and non fixed key for extraction
        if fixed_key:
            ddb_key = fixed_key
        else:
            if "classification" in event and "imageType" in event["classification"]:
                ddb_key = event["classification"]["imageType"]
                logger.debug(f"image_type: {ddb_key}")
            else:
                raise ValueError(
                    f"no [classification][imageType] given in event: {event}"
                )

        # Load classification prompt from DDB
        try:
            ddb_prompt_entry: BedrockPromptConfig = BedrockPromptConfig.get(ddb_key)
        except DoesNotExist:
            raise ValueError(f"no DynamoDB item with key: '{ddb_key}' was found")
        prompt_template = ddb_prompt_entry.prompt_template

        # Load source image from S3
        if (
            "manifest" in event
            and "s3Path" in event["manifest"]
        ):
            image_path = event["manifest"]["s3Path"]
        else:
            raise ValueError(
                "no ['manifest']['s3Path'] to get the text file from "
            )

        image = get_image_file_from_s3(s3_path=image_path).decode('utf-8')

        logger.debug(prompt_template)
        
        mime_type = ""
        # Get mime type
        if "mime" in event:
            mime_type = event["mime"]
        
        # Configure prompt
        classification_message_config = [
            {"role": "user", 
             "content": [
                 {"type": "image","source": { "type": "base64","media_type": mime_type,"data": image}},
                 {"type": "text", "text": prompt_template}
             ]}
        ]
        
        response = generate_message(
            bedrock_runtime=bedrock_rt, model_id=bedrock_model_id, messages=classification_message_config, max_tokens=512, temp=0.5, top_p=0.9
        )
        
        if "completion" in response:
            output_text = response['completion']
        elif "content" in response:
            output_text = response['content'][0]['text']
            logger.debug(output_text)
        else:
            output_text = ""

        logger.debug(response)
        
        if ddb_key == "CLASSIFICATION":
            classification_json = json.loads(output_text)
            image_type = classification_json['CLASSIFICATION']
            
            classification_dict = {"imageType": image_type}
            event['classification'] = classification_dict
            # event.setdefault('classification', {})['imageType'] = image_type

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