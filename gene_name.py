import requests

def get_gene_symbol(ensembl_id):
    # Remove version suffix if present (e.g., ".7" in "ENSG00000244694.7")
    base_ensembl_id = ensembl_id.split(".")[0]
    url = f"https://rest.ensembl.org/lookup/id/{base_ensembl_id}?content-type=application/json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the response JSON
        data = response.json()
        
        # Return the gene symbol if available
        return data.get("display_name", "Gene symbol not found")
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching gene symbol: {e}")
        return None

# Example usage
# ensembl_id = "ENSG00000180105.11"
# gene_symbol = get_gene_symbol(ensembl_id)
# print(f"Gene symbol for {ensembl_id}: {gene_symbol}")
