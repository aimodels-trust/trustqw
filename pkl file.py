import requests

# URL of the .pkl file on Google Drive
file_id = "1e3K7NAowcpfxWc7sfqpk3YHVeyzEDl4J"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Save the file locally
response = requests.get(url)
with open("credit_default_model.pkl", "wb") as file:
    file.write(response.content)

print("Model downloaded successfully!")
