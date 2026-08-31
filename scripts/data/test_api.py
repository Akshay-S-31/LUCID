import requests
url = "https://datasets-server.huggingface.co/rows?dataset=vikhyatk%2Fnyu_depth_v2&config=default&split=train&offset=0&length=2"
res = requests.get(url).json()
if 'rows' in res:
    print("Keys in row:", res['rows'][0]['row'].keys())
else:
    print("Error:", res)
