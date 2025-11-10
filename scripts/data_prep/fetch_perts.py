import requests

def fetch():
    url = "https://pub-19251b2901934f4faa4770fd249554b2.r2.dev/ESM2_pert_features_ensembl_22631.pt"
    output_file = "ESM2_pert_features_ensembl_22631.pt"

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {output_file}")
