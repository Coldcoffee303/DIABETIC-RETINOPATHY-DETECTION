from flask import Flask, request, jsonify, render_template
from model import DRGNN, Predict
import torch


app = Flask(__name__)

# Load model


model = DRGNN(in_channels=2048, hidden_channels=128, num_classes=5)
model.load_state_dict(torch.load("dr_gnn_model.pth", map_location=torch.device('cpu')))
model.eval()



#-----------------------------------------------------------------------------------------------------------

@app.route('/')
def home():
    title = 'Pest - Home'
    return render_template('index.html', title=title)


@app.route('/predict', methods=["GET"])
def predictionform():
    return render_template('form.html')

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["file"]
    image_path = "temp.png"
    image.save(image_path)
    
    pred = Predict(model, image_path)
    

    dr_stages = {
        0: {
            "name": "No Diabetic Retinopathy",
            "description": "No visible signs of diabetic retinopathy detected in the retina.",
            "symptoms": "No symptoms specific to diabetic retinopathy, but regular eye check-ups are still recommended.",
            "effects": "No negative effects on vision due to diabetic retinopathy at this stage.",
            "treatment": "Continue regular diabetes management and schedule annual eye examinations.",
            "risk": "low",
            "next_steps": "Maintain good blood sugar control and have regular eye check-ups."
        },
        1: {
            "name": "Mild Non-Proliferative Diabetic Retinopathy (NPDR)",
            "description": "Early stage of diabetic retinopathy with small areas of balloon-like swelling in the retina's tiny blood vessels.",
            "symptoms": "Usually no noticeable symptoms, though some patients may experience slightly blurred vision.",
            "effects": "Minimal impact on vision at this stage, but indicates that diabetes is affecting eye health.",
            "treatment": "No specific eye treatment is typically needed, but improved diabetes management is essential.",
            "risk": "moderate",
            "next_steps": "Improve blood sugar control and have eye examinations every 9-12 months."
        },
        2: {
            "name": "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)",
            "description": "Progression of the disease with more blood vessels affected, leading to decreased blood flow to the retina.",
            "symptoms": "Blurred vision, fluctuating vision, impaired color vision, or floaters may begin to appear.",
            "effects": "Blood vessels may swell, distort vision, and occasionally leak fluid into the retina.",
            "treatment": "Careful monitoring is required, and some cases may need laser treatment if macular edema is present.",
            "risk": "high",
            "next_steps": "More frequent eye examinations (every 6-8 months) and aggressive diabetes management."
        },
        3: {
            "name": "Severe Non-Proliferative Diabetic Retinopathy (NPDR)",
            "description": "Advanced stage with significant blockage of blood vessels, depriving the retina of proper blood supply.",
            "symptoms": "More noticeable vision problems, including blurred vision, dark or empty areas in vision, and difficulty with night vision.",
            "effects": "The retina signals for new blood vessel growth due to poor blood supply, leading to a high risk of progression.",
            "treatment": "Laser treatment (panretinal photocoagulation) may be recommended to prevent progression to proliferative DR.",
            "risk": "very high",
            "next_steps": "Frequent eye examinations (every 3-4 months) and immediate consultation with a retina specialist."
        },
        4: {
            "name": "Proliferative Diabetic Retinopathy (PDR)",
            "description": "Most advanced stage where new abnormal blood vessels grow on the surface of the retina or optic nerve.",
            "symptoms": "Severe vision loss, floating spots, blurred vision, dark areas, and vision loss that can progress to blindness if untreated.",
            "effects": "New blood vessels are fragile and can leak blood into the eye (vitreous hemorrhage) or cause retinal detachment.",
            "treatment": "Urgent laser treatment, intravitreal injections, or vitrectomy surgery may be required.",
            "risk": "critical",
            "next_steps": "Immediate treatment is necessary. Regular follow-ups with a retina specialist are crucial."
        }
    }
    

    stage_info = dr_stages.get(pred, dr_stages[0]) 
    
    return render_template("result.html", prediction=pred, stage_info=stage_info)

if __name__ == "__main__":
    app.run(debug=True)
