import glob
import os
from multiprocessing.dummy import Pool as ThreadPool

import boto3


def upload_file(localpath, s3path):
    """
Uploads a file to s3
    :param localpath: The local path
    :param s3path: The s3 path in format s3://mybucket/mydir/mysample.txt
    """

    bucket, key = get_bucketname_key(s3path)

    if key.endswith("/"):
        key = "{}{}".format(key, os.path.basename(localpath))

    s3 = boto3.client('s3')

    s3.upload_file(localpath, bucket, key)


def get_bucketname_key(uripath):
    assert uripath.startswith("s3://")

    path_without_scheme = uripath[5:]
    bucket_end_index = path_without_scheme.find("/")

    bucket_name = path_without_scheme
    key = "/"
    if bucket_end_index > -1:
        bucket_name = path_without_scheme[0:bucket_end_index]
        key = path_without_scheme[bucket_end_index + 1:]

    return bucket_name, key


def download_file(s3path, local_dir):
    bucket, key = get_bucketname_key(s3path)

    s3 = boto3.client('s3')

    local_file = os.path.join(local_dir, s3path.split("/")[-1])

    s3.download_file(bucket, key, local_file)

    return local_file


def download_object(s3path):
    bucket, key = get_bucketname_key(s3path)

    s3 = boto3.client('s3')

    s3_response_object = s3.get_object(Bucket=bucket, Key=key)
    object_content = s3_response_object['Body'].read()

    return len(object_content)


def list_files(s3path_prefix):
    assert s3path_prefix.startswith("s3://")
    assert s3path_prefix.endswith("/")

    bucket, key = get_bucketname_key(s3path_prefix)

    s3 = boto3.resource('s3')

    bucket = s3.Bucket(name=bucket)

    return ((o.bucket_name, o.key) for o in bucket.objects.filter(Prefix=key))


def upload_files(local_dir, s3_prefix, num_threads=20):
    input_tuples = ((f, s3_prefix) for f in glob.glob("{}/*".format(local_dir)))

    with ThreadPool(num_threads) as pool:
        pool.starmap(upload_file, input_tuples)


def download_files(s3_prefix, local_dir, num_threads=20):
    input_tuples = (("s3://{}/{}".format(s3_bucket, s3_key), local_dir) for s3_bucket, s3_key in list_files(s3_prefix))

    with ThreadPool(num_threads) as pool:
        results = pool.starmap(download_file, input_tuples)


def download_objects(s3_prefix, num_threads=20):
    s3_files = ("s3://{}/{}".format(s3_bucket, s3_key) for s3_bucket, s3_key in list_files(s3_prefix))

    with ThreadPool(num_threads) as pool:
        results = pool.map(download_object, s3_files)

    return sum(results) / 1024


def get_directory_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def get_s3file_size(bucket, key):
    s3 = boto3.client('s3')
    response = s3.head_object(Bucket=bucket, Key=key)
    size = response['ContentLength']
    return size


def download_files_min_files(s3_prefix, local_dir, min_file_size=310, num_threads=20):
    input_tuples = (("s3://{}/{}".format(s3_bucket, s3_key), local_dir) for s3_bucket, s3_key in list_files(s3_prefix)
                    if get_s3file_size(s3_bucket, s3_key) > min_file_size)

    with ThreadPool(num_threads) as pool:
        results = pool.starmap(download_file, input_tuples)
