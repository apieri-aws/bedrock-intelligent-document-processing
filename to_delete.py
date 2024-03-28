import boto3
import json
from pdf2image import convert_from_bytes
from typing import Tuple
import base64

s3 = boto3.client("s3")

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
    return o.get("Body").read()
    # return base64.b64encode(o.get("Body").read())

s3path = "s3://bedrockidpclaude3workflow-bedrockidpclaude3bucket0-pgrdbpkwtsfg/uploads/unc_paystub_5.pdf"


image_file = get_image_file_from_s3(s3path)

processed_image = convert_from_bytes(image_file, fmt="png")

s3.put_object(Body=processed_image, Bucket='bedrockidpclaude3workflow-bedrockidpclaude3bucket0-pgrdbpkwtsfg', Key='processed.png')