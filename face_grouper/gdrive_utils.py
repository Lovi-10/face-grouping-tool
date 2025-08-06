from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import re

def extract_folder_id_from_url(url):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else url

def download_gdrive_folder(folder_url_or_id, dest, progress_callback=None):
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive

    folder_id = extract_folder_id_from_url(folder_url_or_id)
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile("client_secret_495485308500-k25o2ciaqpt2dcm7k21hsq83b9732e4g.apps.googleusercontent.com.json")
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    os.makedirs(dest, exist_ok=True)

    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    total_files = len(file_list)

    paths = []
    for i, file in enumerate(file_list):
        filename = os.path.join(dest, file['title'])
        file.GetContentFile(filename)
        paths.append(filename)

        # Update progress if callback is provided
        if progress_callback:
            progress_callback((i + 1) / total_files)

    return paths

