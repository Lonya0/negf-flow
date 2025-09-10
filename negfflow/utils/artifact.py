from dflow import (
    S3Artifact,
    s3_config,
    upload_artifact,
)

def get_artifact_from_uri(uri):
    if uri.startswith("s3://"):
        return S3Artifact(uri[5:])
    if uri.startswith("oss://"):
        return S3Artifact(uri[6:])
    raise ValueError(f"Unrecognized scheme of URI: {uri}")

def upload_artifact_and_print_uri(files, name):
    art = upload_artifact(files)
    if s3_config["repo_type"] == "s3" and hasattr(art, "key"):
        print(f"{name} has been uploaded to s3://{art.key}")
    elif s3_config["repo_type"] == "oss" and hasattr(art, "key"):
        print(f"{name} has been uploaded to oss://{art.key}")
    return art
