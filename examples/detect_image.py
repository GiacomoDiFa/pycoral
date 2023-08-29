# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to detect objects in a given image.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh detect_image.py

python3 examples/detect_image.py \
  --model test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels test_data/coco_labels.txt \
  --input test_data/grace_hopper.bmp \
  --output ${HOME}/grace_hopper_processed.bmp
```
"""

import argparse
import time
import glob
import json

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from collections import defaultdict

# Carica il file JSON
with open('/home/mendel/research_coral/pycoral/assets/labels/annotations.json', 'r') as file:
    data = json.load(file)

# Crea un dizionario in cui archiviare il conteggio dei category_id
category_count = defaultdict(int)

# Itera attraverso le voci nel file JSON e conta i category_id
for entry in data:
    segments_info = entry['segments_info']
    for segment_info in segments_info:
        category_id = segment_info['category_id']
        category_count[category_id] += 1

sorted_category_count = dict(sorted(category_count.items(), key=lambda item: item[0]))

# Stampa i risultati
for category_id, count in sorted_category_count.items():
    print(f"Di category_id {category_id} ne ho trovati {count}")

def count_y_pred(y_pred):
  conteggio = {}
  for numero in y_pred:
    if numero in conteggio:
      conteggio[numero] += 1
    else:
      conteggio[numero] = 1
  return conteggio

  


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file')
  parser.add_argument('-i', '--input',
                      help='File path of image to process')
  parser.add_argument('-l', '--labels', help='File path of labels file')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations')
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  args = parser.parse_args()

  labels = read_label_file(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  image_list = []
  scale_list = []
  objs_list = []
  y_pred = []

  for filename in glob.glob('/home/mendel/research_coral/pycoral/assets/original_images/*.jpg'):
    im = Image.open(filename)
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    _, scale = common.set_resized_input(interpreter, im.size, lambda size: im.resize(size, Image.ANTIALIAS))
    
    print('----INFERENCE TIME----')
    for _ in range(args.count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        print('%.2f ms' % (inference_time * 1000))
        
        
    objs = detect.get_objects(interpreter, args.threshold, scale)
    objs_list.append(objs)
  #print(objs_list)
  print('-------RESULTS--------')

  for i in range(len(objs_list)):
    if not objs_list[i]:
      print('No objects detected in image: ', i)
    else:
      print('Objects detected in image', i)
      for j in range(len(objs_list[i])):
        print(labels.get(objs_list[i][j].id, objs_list[i][j].id))
        print('  id:    ', objs_list[i][j].id)
        y_pred.append(objs_list[i][j].id)
        print('  score: ', objs_list[i][j].score)
        print('  bbox:  ', objs_list[i][j].bbox)
  y_pred.sort()
  conteggio_y_pred = count_y_pred(y_pred)
  print(conteggio_y_pred)
  print(sorted_category_count)
  for key in sorted_category_count:
    if key not in conteggio_y_pred:
      conteggio_y_pred[key] = 0
  conteggio_y_pred_sorted = dict(sorted(conteggio_y_pred.items(), key=lambda item: item[0]))
  print(conteggio_y_pred_sorted)
  if args.output:
    image = image.convert('RGB')
    draw_objects(ImageDraw.Draw(image), objs, labels)
    image.save(args.output)
    image.show()


if __name__ == '__main__':
  main()