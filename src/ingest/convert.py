from pathlib import Path



'''
BDD100K label JSON structure (one entry per image, 70k entries in the train file):

{
  "videoName": "0000f77c-6257be58",
  "name": "0000f77c-6257be58.jpg",
  "labels": [
    {
      "category": "car",
      "box2d": {
        "x1": 49.44,
        "y1": 254.53,
        "x2": 357.81,
        "y2": 487.91
      },
      "attributes": { "occluded": false, "truncated": false }
    },
    {
      "category": "traffic light",
      "box2d": {
        "x1": 1125.90,
        "y1": 133.18,
        "x2": 1156.98,
        "y2": 210.88
      },
      "attributes": { "occluded": false, "truncated": false }
    }
  ],
  "attributes": { "weather": "clear", "timeofday": "daytime", "scene": "city street" }
}

YOLO .txt output format (one file per image, one line per box):
  class_id  x_center  y_center  width  height
All values normalized to [0, 1]. Image size is 1280x720.

Categories (10 classes):
  0: pedestrian, 1: rider, 2: car, 3: truck, 4: bus,
  5: train, 6: motorcycle, 7: bicycle, 8: traffic light, 9: traffic sign
'''




def convert(label_path: Path, yolo_dir: Path):
	# BDD100K JSON → YOLO .txt labels
	'''
	label_path: input to json labels
	yolo_dir: output to yolo txt labels
	'''
	categories = json.loads(Path("configs/categories.json").read_text())

	# open the json
	try:
		with label_path.open("r", encoding="utf-8") as file:
			data = json.load(file)
		
	except FileNotFoundError:
		print("Error: File not found.")
		return


	'''
	How yolo expects the data

	class_id x_center y_center width height                                                                                                                        
	                                                                                                                                                         
	So for a car (class_id=2) with box2d: {x1: 49.44, y1: 254.53, x2: 357.81, y2: 487.91} on a 1280x720 image:                                                     
	                                                                                                                                                       
	2 0.1591 0.5153 0.2409 0.3241                                                                                                                                  
	                                                                                                                                                         
	Where:
	- x_center = (49.44 + 357.81) / 2 / 1280                                                                                                                       
	- y_center = (254.53 + 487.91) / 2 / 720                                                                                                                       
	- width = (357.81 - 49.44) / 1280       
	- height = (487.91 - 254.53) / 720   


	'''


	# iterate over the data dictionary
	for video in data:
		label_file = Path(videoName + ".txt")
		labels = video["labels"]
		with label_file.open("a") as file:
			
			header = 
			file.write()
			for label in labels:
				# process each label and bbox -> write it to the file
				



	return