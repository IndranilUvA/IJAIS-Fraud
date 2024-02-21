import requests
url = "https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py"

response = requests.get(url)

if response.status_code == 200:
    with open("tokenization.py", "wb") as file:
        file.write(response.content)
    print("File downloaded successfully.")
else:
    print("Failed to download file. Status code:", response.status_code)