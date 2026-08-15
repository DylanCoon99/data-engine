from pathlib import Path
import json



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




def convert(data: DataConfig):
	# BDD100K JSON → YOLO .txt labels
	'''
	label_path: input to json labels
	yolo_dir: output to yolo txt labels
	'''
	categories = json.loads(Path("configs/categories.json").read_text())

	label_path = data.raw_dir / "labels" / "det_v2_train_release.json"
	yolo_dir = data.yolo_dir

	image_width = data.image_width
	image_height = data.image_height

	# open the json
	try:
		with label_path.open("r", encoding="utf-8") as file:
			raw_labels = json.load(file)
		
	except FileNotFoundError:
		print("Error: File not found.")
		return

	# iterate over the data dictionary
	for video in raw_labels:
		videoName = video["videoName"]
		label_file = yolo_dir / (videoName + ".txt")
		labels = video["labels"]
		with label_file.open("w") as file:
			for label in labels:
				# process each label and bbox -> write it to the file
				class_id = categories[label["category"]]
				bbox = label["box2d"]
				x1, x2, y1, y2 = bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]
				x_center = (x1 + x2) / 2 / image_width
				y_center = (y1 + y2) / 2 / image_height
				width = (x2 - x1) / image_width
				height = (y2 - y1) / image_height
				file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

	return