# import clarifai
# import clarifai.rest as TT
from clarifai.rest import ClarifaiApp

app = ClarifaiApp()

model = app.models.get('general-v1.3')
response = model.predict_by_url(url='https://samples.clarifai.com/metro-north.jpg')

concepts = response['outputs'][0]['data']['concepts']
for concept in concepts:
    print(concept['name'], concept['value'])


