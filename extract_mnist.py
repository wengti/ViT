from pathlib import Path
import _csv as csv
import numpy as np
import cv2

def extract_mnist(csv_fname, save_dir):
    
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        save_dir.mkdir(parents = True,
                       exist_ok = True)
    
    with open(csv_fname, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            
            if idx == 0:
                continue
            
            img = np.zeros(28*28)
            img[:] = list(map(int, row[1:]))
            img = img.reshape((28,28))
            
            save_class = row[0]
            save_class_folder = save_dir / save_class
            if not save_class_folder.is_dir():
                save_class_folder.mkdir(parents = True,
                                        exist_ok = True)
            
            save_name = f"{idx}.png"
            save_file = save_class_folder / save_name
            cv2.imwrite(save_file, img)
            
            if (idx+1) % 1000 == 0:
                print(f"[INFO] {idx+1} images have been saved into {save_dir}.")

extract_mnist(csv_fname = "./mnist_train.csv",
              save_dir = "./train/images")

extract_mnist(csv_fname = "./mnist_test.csv",
              save_dir = "./test/images")

