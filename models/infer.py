
from ultralytics import YOLO
from ultralytics.utils import ops
import torch
import numpy as np
from openvino import Core 
import openvino.runtime as ov
import onnxruntime
import cv2
from abc import ABC , abstractmethod
from pathlib import Path
import os.path as osp
from models.utils.box import letterbox
from typing import Tuple , List
'''
Preprocesing : 
Image to Array

Detect : 
Complete From Preprocessing to Postprocessing

Post Processing : 
After detection such as nms and draw BBOX

def __call__ :
Only do Detect


'''
class Inference: 
    def __new__(*args , **kwargs):
        
        kwargs = kwargs['kwargs']
        weight_path = Path("./models/weights").resolve()
        inference_engine = kwargs["inference_engine" ]
        format_ = {"onnx" : ".onnx" , "torch" : ".pt" , "openvino" : ""}
        model_versioin  = kwargs["model_version" ]
        input_sz = kwargs["imgsz" ]
        inference_version = kwargs["inference_version"]

        model_path = osp.join(weight_path , inference_engine  , model_versioin , input_sz , inference_version + format_[inference_engine] )
        
        # Check for Path 

        if not osp.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        INFERENCE_ENGINES = {
            "torch": YOLOInference,
            "onnx": ONNXInference,
            "openvino": OpenVinoInference
        }
        
        engine_class = INFERENCE_ENGINES.get(inference_engine)

        if engine_class is not None:
            return engine_class(model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported inference engine: {inference_engine}")
class InferModel(ABC):
    '''
    
    If we want to use different formats of model with Ultralytics Inference
    
    '''

    def __init__(self , model_path : str, *args , **kwargs):
        ## 
        self.model_path = model_path    

        ## Configs
        self.confidence_thres = kwargs.get("confidence_thres" , 0.6)
        self.iou_thres = kwargs.get("iou_thres" , 0.5)
        self.image_size = kwargs.get("imgsz" , "224")
        ## Get CLasses 
        self.classes = {0 : "Charger" , 1 : "Background"}
        # Load the class names from the COCO dataset

        # Generate a color palette for the classes
        self.color_palette = np.random.randint(0, 255, size=(len(self.classes), 3) , dtype= np.uint8)
        # Convert to tuple
    @abstractmethod
    def detect(self , image):
        """
        Every Type of infeernce must have this abstract that returns the following :

        1 . results : Image output of tf the detection
        2. loc : Location of the detections / bbox
    
        """
        pass
    @abstractmethod
    def preprocess(self , image):
        pass


    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        loc = []
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            loc.append((box , score , class_id))
            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)
        # Return the modified input image
        return input_image ,loc

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]
        color = color.tolist()
        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def __call__(self,x) :
        return self.detect(x)
    


### Torchscript Inference from Ultralytics

class YOLOInference(InferModel):
    def __init__(self , model_path : str , *args , **kwargs):
        
        super().__init__(model_path)
        self.model = YOLO(task="detect" , model=self.model_path , **kwargs)
    def detect(self , image):
        
        results = self.model(image)
        results = results[0]
        annotated_frame = results.plot()
        return annotated_frame , results.bbox.xyxy


# Annotate the image with results


### ONNX
class ONNXInference(InferModel):
    def __init__(self , model_path , *args , **kwargs):
        
        super().__init__(model_path)
        self.confidence_thres = kwargs.get("confidence_thres" , 0.7)
        self.iou_thres = kwargs.get("iou_thres" , 0.7)

        self.session = onnxruntime.InferenceSession(model_path , providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.model_inputs = self.session.get_inputs()
        self.input_shape = self.model_inputs[0].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]


    def detect(self , image):
        image_data = self.preprocess(image)
        output = self.session.run(None , {self.model_inputs[0].name : image_data})
        results , loc = self.postprocess(image , output)
        return  results , loc

    def preprocess(self , image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV

        # Get the height and width of the input image
        self.img_height, self.img_width = image.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(image  , cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data
    

    



class OpenVinoInference(InferModel):

    '''
    OpenVino Inference
    
    A fixed size Input to reduce the inference time when reshaping
    '''
    def __init__(self , model_path : str , *args , **kwargs):
        super().__init__(model_path , *args , **kwargs)
        self.core = Core()
        self.model = self.core.read_model(model = osp.join(model_path , "best.xml") , weights = osp.join(model_path , "best.bin"))    
        print(f"CPU Device : {self.core.available_devices}")
        ## Model Input Shape

        ## Get Image Inpt Shape during inference

        self.input_shape = self.model.input(0).shape
        self.input_width = self.input_shape[3]
        self.input_height = self.input_shape[2]


        self.img_width = self.input_width
        self.img_height = self.input_height



        self.compiled_model = self.core.compile_model(self.model , "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
    def preprocess(self , image):
    
        # resize
        
        img = letterbox(image, auto=False , new_shape= (int(self.input_height) , int(self.input_width)) )[0]
        
        # Convert
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        ## Transform fn
        input_tensor = img.astype(np.float32)  # uint8 to fp16/32
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor



    def detect(self , image ):

        if isinstance(image, np.ndarray):
            img = image
        else:
            img = cv2.imread(str(image))
        
        preprocessed_img = self.preprocess(img)
        # Model
        output =  self.compiled_model(preprocessed_img)[0]
        # Post Processing
        results , loc = self.postprocess(image , output)
        return results , loc
        # postprocessing =




'''
NMS OpenVINO vs ONNX
Outputs : 
Shape:

OpenVINO NMS
(1 ,5 ,1029)



'''