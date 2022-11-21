import tensorflowjs as tfjs

model = tfjs.converters.load_keras_model('../saved_model/model-20221121.json', '../saved_model')
print(model)