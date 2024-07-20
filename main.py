import copy
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageDraw 

from HuTuTu import TuTu


colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red']


def plot_bbox(image, data):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.show()


def draw_ocr_bboxes(image, prediction, scale=1):
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                    "{}".format(label),
                    align="right",
                    fill=color)
       
    image.show()


if __name__ == '__main__':
    tutu = TuTu()

    # 图像任务
    image_path = "tests/images/car.jpg"
    image = Image.open(image_path)

    # Caption
    caption = tutu.caption(image)
    print(f"caption: {caption}")
    # Detailed Caption
    detailed_caption = tutu.detailed_caption(image)
    print(f"detailed_caption: {detailed_caption}")
    # More detailed caption
    more_detailed_caption = tutu.more_detailed_caption(image)
    print(f"more_detailed_caption: {more_detailed_caption}")

    # Object detection
    object_detection = tutu.object_detection(image)
    print(f"object_detection: {object_detection}")
    plot_bbox(image, object_detection["<OD>"])
    # Dense region caption
    dense_region_caption = tutu.dense_region_caption(image)
    print(f"dense_region_caption: {dense_region_caption}")
    plot_bbox(image, dense_region_caption["<DENSE_REGION_CAPTION>"])
    # Region proposal
    region_proposal = tutu.region_proposal(image)
    print(f"region_proposal: {region_proposal}")
    plot_bbox(image, region_proposal["<REGION_PROPOSAL>"])

    # Phrase Grounding
    phrase_grounding = tutu.phrase_grounding(image, "A green car parked in front of a yellow building.")
    print(f"phrase_grounding: {phrase_grounding}")
    plot_bbox(image, phrase_grounding["<CAPTION_TO_PHRASE_GROUNDING>"])

    # OCR 任务
    image_path = "tests/images/ocr_test_case.png"
    image = Image.open(image_path)

    # OCR
    ocr = tutu.ocr(image)
    print(f"ocr: {ocr}")
    # OCR wth Region
    ocr_with_region = tutu.ocr(image)
    print(f"ocr_with_region: {ocr_with_region}")
    output_image = copy.deepcopy(image)
    w, h = output_image.size
    scale = 800 / max(w, h)
    new_output_image = output_image.resize((int(w * scale), int(h * scale)))
    draw_ocr_bboxes(new_output_image, ocr_with_region['<OCR_WITH_REGION>'], scale=scale)
