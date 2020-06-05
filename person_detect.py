
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        """
        returning num_people
        """
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        
        self.exec_net = None


        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        core= IECore()
        self.exec_net = core.load_network(self.model, self.device, num_requests = 1)
        
        #raise NotImplementedError
        
    def predict(self, image):
        #print("Inside predict functio :\nShape of input model", self.input_shape)
        #Shape of input model [1, 3, 320, 544]
        #print("shape of frame: ", image.shape)
        #shape of frame:  (1080, 1920, 3)
        height = image.shape[0]
        width = image.shape[1]
        img = self.preprocess_input(image)
        #print("img shape:", img.shape)
        #img shape: (1, 3, 320, 544)
        result = self.exec_net.infer({self.input_name: img})
        #print("After sync request --\nresult :", list(result.keys()))
        #result : ['detection_out']
        #print("detection out shape is :", result["detection_out"].shape)
        #detection out shape is : (1, 1, 200, 7)
        coords = self.preprocess_outputs(result['detection_out'], width, height)
#         print("after self.preprocess output:\ncoords shape is ", len(coords))
#         print("Coord explore\n",coords)
        img = self.draw_outputs(coords, image)
        return (coords, img)
        
        #raise NotImplementedError
    
    def draw_outputs(self, coords, image):
        for coord in coords:
            cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255))
        return image
        
        #raise NotImplementedError

    def preprocess_outputs(self, outputs, width, height):
        coords = []
       
        for output in outputs[0][0]:
            if output[1] == 1 and output[2] >= self.threshold:
                xmin = int(output[3] * width)
                ymin = int(output[4] * height)
                xmax = int(output[5] * width)
                ymax = int(output[6] * height)
                coords.append([xmin, ymin, xmax, ymax])
        return coords
                
        #raise NotImplementedError

    def preprocess_input(self, image):
        p_img = cv2.resize(image.copy(), (self.input_shape[3], self.input_shape[2]))
        p_img = p_img.transpose((2, 0, 1))
        p_img = p_img.reshape(1, *p_img.shape)
        
        return p_img
        
        
        #raise NotImplementedError


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)  # returns input array from the disk with npy extension
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)