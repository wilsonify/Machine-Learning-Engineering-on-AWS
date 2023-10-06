import fnmatch

import boto3

s3_bucket = "064592191516-ml-engineering"


def glob_s3(glob_pattern):
    s3_client = boto3.client('s3')
    first_asterisk_index = glob_pattern.find('*')
    s3_prefix = glob_pattern[:first_asterisk_index] if first_asterisk_index >= 0 else glob_pattern
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    matched_objects = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        is_matched = fnmatch.fnmatch(key, glob_pattern)
        if is_matched:
            matched_objects.append(key)
    return matched_objects


training_samples = glob_s3(f"c06/train/**/*.png")
n_training_samples = len(training_samples)
print(f"n_training_samples = {n_training_samples}")

for i in range(0, 10):
    image_files = glob_s3(f"c06/train/{i}/*.png")
    print(f'---{i}---')
    print(f"image_files = {image_files}")
    # ipyplot.plot_images(image_files, max_images=5, img_width=128)
