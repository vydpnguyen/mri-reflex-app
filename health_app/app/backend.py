 #change flask to reflex
from flask import Flask, request, render_template, redirect, url_for

from werkzeug.utils import secure_filename
import os

import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './image_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# use singleton class to call infer function
# Model Singleton
class ModelSingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if ModelSingleton._instance is None:
            ModelSingleton()
        return ModelSingleton._instance

    def __init__(self):
        if ModelSingleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ModelSingleton._instance = self
            self.device = "cpu"
            self.model = torchvision.models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, 4)  # Adjust for 4 classes
            self.model.load_state_dict(torch.load("/home/health_app/checkpoints/checkpoint_epoch_4.pth", map_location=self.device)['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            # Define class names manually or load from a file
            self.class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    def infer(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class_idx = predicted.item()
        # Return the class name instead of the index
        return self.class_names[predicted_class_idx]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            predicted_class = ModelSingleton.get_instance().infer(filepath)
            return render_template('result.html', predicted_class=predicted_class)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
