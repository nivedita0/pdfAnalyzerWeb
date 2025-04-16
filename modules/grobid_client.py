import requests

def process_pdf_with_grobid(pdf_path):
    url = "http://localhost:8070/api/processFulltextDocument"
    with open(pdf_path, 'rb') as f:
        files = {'input': (pdf_path, f)}
        response = requests.post(url, files=files)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error: {response.status_code}")
            return None
