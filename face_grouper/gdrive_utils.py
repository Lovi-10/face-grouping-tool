from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import re

def extract_folder_id_from_url(url):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else url

def download_gdrive_folder(folder_url_or_id, dest):
    folder_id = extract_folder_id_from_url(folder_url_or_id)
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    os.makedirs(dest, exist_ok=True)

    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    paths = []
    for file in file_list:
        filename = os.path.join(dest, file['title'])
        file.GetContentFile(filename)
        paths.append(filename)

    return paths