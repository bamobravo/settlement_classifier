import cv2
import numpy as np


def detectSettlement(image):
	cv2.imread(image)
	width = image.shape[1]
	height = image.shape[0]
	scale = 0.00392
	classes =['large','average','low']
	colors = np.random.uniform(0,255,size=(len(classes),3))
	weight =''
	config =''
	net =cv2.dnn.readnet(weight,config)
	blob = cv2.dnn.blobFromImage(image,scale, (416,416), (0,0,0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(get_output_layers(net))
	class_ids = []
	confidences = []
	boxes = []
	conf_threshold = 0.5
	nms_threshold = 0.4

	for out in outs:
	    for detection in out:
	        scores = detection[5:]
	        class_id = np.argmax(scores)
	        confidence = scores[class_id]
	        if confidence > 0.5:
	            center_x = int(detection[0] * Width)
	            center_y = int(detection[1] * Height)
	            w = int(detection[2] * Width)
	            h = int(detection[3] * Height)
	            x = center_x - w / 2
	            y = center_y - h / 2
	            class_ids.append(class_id)
	            confidences.append(float(confidence))
	            boxes.append([x, y, w, h])
	

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)