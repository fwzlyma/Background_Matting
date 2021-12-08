import cv2
import numpy as np


mask = cv2.imread(r"F:\Code\projects\Background-Matting-master\sample_data\input\0001_masksDL.png")
print(mask)
where = np.array(np.where(mask))
print(where)