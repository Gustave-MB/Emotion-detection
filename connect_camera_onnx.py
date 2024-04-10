import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import  onnx
import onnxruntime as ort


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_numpy(X):
   """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        X (torch.Tensor): Input PyTorch tensor.

    Returns:
        numpy.ndarray: NumPy array representation of the tensor.
    """
   return X.detach().cpu().numpy() if X.requires_grad else  X.cpu().numpy()

path = 'FER2013.onnx'
onnx.checker.check_model(onnx.load(path))
ort_session = ort.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name


valid_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])


# Connects to your computer's default camera
cap = cv2.VideoCapture(0)

# # Automatically grab width and height from video feed
# # (returns float which we need to convert to integer for later on!)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    img = np.zeros((48,48,3), dtype=np.uint8)
    img[:,:,0] = gray  # Assign the grayscale values to the red channel
    img[:,:,1] = gray  # Assign the grayscale values to the green channel
    img[:,:,2] = gray # Assign the grayscale values to the blue channel
    img = img.astype('uint8')

    # Apply transforms and and unsqueze
    img_tensor = valid_transforms(img).unsqueeze(0)


    preds = ort_session.run(None, {input_name: to_numpy(img_tensor)})
    preds = torch.squeeze(torch.from_numpy(np.array(preds)),0)
    
    # preds = model(img_tensor)
    # model.eval()
    prob = F.softmax(preds, dim=1)
    

    #find the index with the greatest probability, set that as predicted label
    prediction = prob.argmax(dim=1, keepdim=True)
    # print(prediction)
    print(classes[prediction])
    # Display the resulting frame
    cv2.imshow('frame',gray)
    
    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()