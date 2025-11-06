import urllib.request

# The model weights of intra coding come from CompressAI.
root_url = "https://compressai.s3.amazonaws.com/models/v1/"

model_names = [
           "cheng2020-anchor-3-e49be189.pth.tar",
           "cheng2020-anchor-4-98b0b468.pth.tar",
           "cheng2020-anchor-5-23852949.pth.tar",
           "cheng2020-anchor-6-4c052b1a.pth.tar",
]

for model in model_names:
    print(f"downloading {model}")
    urllib.request.urlretrieve(root_url+model, model)
