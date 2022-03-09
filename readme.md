An easy and fast way to serve complex machine learning inference graphs at scale on Kubernetes.

Moving machine learning models from lab to production is not an easy task. In fact, in many ML projects, it is as challenging as solving the actual data modelling problem or even harder, especially when thousands or millions of inference hits are expected to be served. In this blog, we will see how deploying machine learning can be easier with Seldon. As a walkthrough example, we will be using Seldon to build the inference graph of an AI OCR that reads Checks. The focus is on server model deployment type.

---
**NOTE**
The prerequisites to this article:
 - Basic knowledge of Docker.
 - Basic knowledge of Kubernetes.
 -  An installed Kubernetes cluster with Istio ingress.
 - Installing [Seldon operator](https://docs.seldon.io/projects/seldon-core/en/latest/examples/seldon_core_setup.html) on Kubernetes.
 - The code is on https://github.com/BoundlessView/seldon-deployment-ai-ocr
---

_*Disclaimer: We have no affiliation with or interest vested in Seldon.io. The aim of the article is purely to share our experience at BoundlessView on how we used Seldon.*_

## A brief intro about AI OCR Check reader:

**The main functional requirement:** of this custom OCR is to extract all textual information from check images. For simplicity, in this blog, the task is limited to extracting only the MICR text from check images.

**digram here**
check image >> output in json

**The non-functional requirement:** OCR functionality has to be deployed at scale and exposed as an API.

**Project main challenges:** the images arrive in several layouts or templates, and many of them contain a mixed text of handwritten and printed or written in two languages; English and Arabic. 

**The solution:** two vision problems are tackled as follow:
 - Faster-R CNN is used to detect and locate the fields of interest and identify the language.
 - Custom implementation of a sequence neural network is used to recognise the text.

**The deployment challenge:**
Once we attained the required accuracy at the lab, we thought the problem was resolved. However, as soon as we started thinking about the production, we realised several challenges. We came up with these thoughts and requirements:

- Putting all the models in a monolith service is not a good idea because each model has different hardware requirements.
- The research and development are still going on to enhance accuracy and speed, so the models will be updated frequently and at a different rate. This is another reason support decoupling the system into microservices.
- We need an orchestrator to control data flow between the microservices.
- The caveat for the last requirement is that we don’t want to implement the orchestrator ourselves.
- The final product will be consist of 5 microservices. We need to run these microservices on Kubernetes, but we want to avoid the complexity of creating and managing Kubernetes resources, including deployment, services, virtual services,...etc


The rest of the article illustrates how these requirements have been accomplished by [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/index.html). Seldon Core is an open-source framework that makes it easier and faster to deploy machine learning models at scale on Kubernetes.

#### Serving models:
The idea of serving is to convert the model weights file into something that can respond to requests. Seldon provides solutions to so many cases in this space. The easiest and fastest one is [Prepackaged Model Servers](https://docs.seldon.io/projects/seldon-core/en/latest/servers/overview.html?highlight=%20Prepackaged%20Model%20Servers). You only need to give the model location to get it served and deployed straightway on Kubernetes. In our case, this option is not appropriate. In both ML tasks, the final inference output results from preprocessing the input data and evaluating it on a deep learning model. What suits our case is Seldon Core Language Wrappers that containerise machine learning models and code to produce docker images that are then ready to run and serve requests through either REST or gRPC interfaces.

Before we dive into how the code and models are converted into microservices, we need to clarify a few concepts:

- Seldon core expects us to create a Python class, as an entry point, that implements one of the abstract methods that Seldon Core recognises.
- To decide which method to include in the class, we have first to identify the service or the [component type](https://docs.seldon.io/projects/seldon-core/en/latest/python/python_component.html) of the microservice. 
- Seldon core offers four components; Model, Combiners, Routers and Transformers. The decision on which type to select depends on what the microservice is doing and its location in the inference graph. For example; 
    1. If the service receives an observation and makes a prediction, then the component or service type is MODEL, and the class has to implement the `predict()` method.  
    2. If the service is needed to combine outputs of another two MODEL services, then the component type is COMBINER, and the class has to implement `aggregate()` method.
    3. For further details, [Python Components](https://docs.seldon.io/projects/seldon-core/en/latest/python/python_component.html) and [Graphs](https://docs.seldon.io/projects/seldon-core/en/latest/examples/graph-metadata.html?)
- Our inference graph consists of two services of type MODEL.
 X=x1 >> model1 >> y1=x2 >> model2>>y2=y 

**Model1**: is the text detector and localiser. Its `predict()` method receives the check image and reurns the bounding box of Check MICR, which is used to crop the MICR area.
**Model2**: is the text recogniser. Its `predict()` method receives the cropped image and returns the text.  

#### Building docker images: 

For text dectection task: 
- Python class `src\inference\detection\Serving.py` implments `predict()` method. The model is loaded in the initilization method.
 
- In the Docker file, we have to use one of Seldon's docker base images. 
```dockerfile
# syntax=docker/dockerfile:1.2
FROM seldonio/seldon-core-s2i-python36:1.1.0
ARG EXPERIMENT_NAME
WORKDIR /app
#install dependencies
RUN python3.6 -m pip install pip==19
RUN pip install opencv-python-headless==4.2.0.34 \
                intel_tensorflow==1.14.0 \                
                gast==0.2.2 \                
                pyyaml \                
                git+https://github.com/tensorpack/tensorpack.git@v0.9.8
#copy Python class and model.
COPY models/$EXPERIMENT_NAME/compact.pb /app/compact.pb
COPY models/$EXPERIMENT_NAME/config.yaml /app/config.yaml
COPY src/inference/$EXPERIMENT_NAME/* /app/
CMD seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE
```
At the run time, `seldon-core-microservice` Python wrapper turns Serving.py into a fully operational microservice that receives requests on REST/gRPC interfaces.

```bash
docker build --build-arg EXPERIMENT_NAME='detection' -t boundlessview/detection-infer -f docker/detection/inference.dockerfile ./
docker push  boundlessview/detection-infer
```
For the text recognition task, the same structure is followed as for the detection.
```bash
docker build --build-arg EXPERIMENT_NAME='recognition' -t boundlessview/recognition-infer -f docker/recognition/inference.dockerfile ./
docker push  boundlessview/recognition-infer
```


#### Deployment


**How are we going to run those microservices on Kuberenes?** A possible answer is by creating typical [Kubernetes Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) for each docker image. This is fine but how detector servie is going to connect to recognizer? Each microservice is prepared to evaluate the input data on the loaded ML model and sends result, while in our inference graph the output of detector to be forwarded to the recognizer. A Orchestration is required here to fecilictate this communication. Seldon Core provides this out of the box through [Seldon Deployment CRD](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/overview.html#seldondeployment-crd) (Custom Resource Definition).

The key components are:

- A list of `Predictors`, each with a specification for the number of replicas.

- Each defines a graph and its set of deployments. Multiple predictors is useful when we want to split traffic between a main graph and a canary or for other production rollout scenarios.

- For each predictor a list of `componentSpecs`. Each `componentSpec` is a Kubernetes `PodTemplateSpec` which Seldon will build into a Kubernetes Deployment. Place here the images from your graph and their requirements, e.g. `Volumes`, `ImagePullSecrets`, `Resources Requests` etc.

- A graph specification that describes how our components are joined together.



```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: checks-reader
  namespace: default
  labels:
    app: seldon
spec:
  annotations:
    deployment_version: v1
  name: checks-reader
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: boundlessview/detection-infer:latest
          name: detection-infer
          env:
          - name: API_TYPE
            value: "REST"
          - name: MODEL_NAME
            value: "Serving"
          - name: SERVICE_TYPE
            value: "MODEL"
    - spec:
        containers:
        - image: boundlessview/recognition-infer:latest
          name: recognition-infer
          env:
          - name: API_TYPE
            value: "REST"
          - name: MODEL_NAME
            value: "Serving"
          - name: SERVICE_TYPE
            value: "MODEL"
    graph:
      name: detection-infer
      endpoint:
        type: REST
      type: MODEL
      parameters:
      - name: input_tensor_names
        value: "image"
        type: STRING
      - name: output_tensor_names
        value: "output/boxes"
        type: STRING
      - name: prefix_name
        value: "TDServing"
        type: STRING
      children:
        - name: recognition-infer
          endpoint:
            type: REST
          type: MODEL
          parameters:
          - name: input_tensor_names
            value: "image,input_length"
            type: STRING
          - name: output_tensor_names
            value: "ema/output/labels,ema/output/probs"
            type: STRING
          - name: prefix_name
            value: "TRServing"
            type: STRING
          children: []
    name: main
    replicas: 1
```

A prerequisite to applying this Seldon deployment on Kubernetes is installing Seldon Core Operator. It reads the CRD definition of Seldon Deployment resources applied to the cluster and ensures that all required components like Pods and Services are created. The operator also creates `Orchestrator Orchestrator` responsible for managing the intra-graph traffic. 

As per our graph, the orchestrator intercepts the received request and forwards it to `detection-infer`. It then routes the output of this component to `recognition-infer`. 

>>>diagram>>>

```sh
kubectl apply -f kube\seldon-deployment.yml
```



#### testing