from google.cloud import storage
from deployment.credentials import credentials

def append_data_to_csv_in_gcs(bucket_name, file_path, prompt, sentiment):
    
    storage_client = storage.Client(credentials=credentials)

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)

    with blob.open("r") as file:
        lines = file.readlines()
        lines.append(f'{prompt},{sentiment}\n')

    with blob.open("w") as file:
        file.writelines(lines)
        
