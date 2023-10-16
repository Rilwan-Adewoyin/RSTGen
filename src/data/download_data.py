import os
import io
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload
import tarfile

def download_files_from_folder(folder_id, save_path):
    creds = None
    # # Load existing credentials if they exist
    # if os.path.exists('token.json'):
    #     creds = Credentials.from_authorized_user_file('token.json')

    # Build the Drive API service
    service = build('drive', 'v3', credentials=creds)

    # Get list of files in the folder
    results = service.files().list(q=f"'{folder_id}' in parents", pageSize=1000).execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        for item in items:
            file_id = item['id']
            request = service.files().get_media(fileId=file_id)

            # Download the file
            file_path = os.path.join(save_path, item['name'])
            fh = io.FileIO(file_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Downloaded {item['name']} {int(status.progress() * 100)}%.")

            # Extract if it's a tar.gz
            if file_path.endswith('.tar.gz'):
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(path=save_path)

if __name__ == "__main__":
    FOLDER_ID = '1seqNux3ycMLl-FMqbDRa3F5LhP7r61vG'
    SAVE_PATH = './data/data_files'

    # Create directory if it doesn't exist
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    download_files_from_folder(FOLDER_ID, SAVE_PATH)
