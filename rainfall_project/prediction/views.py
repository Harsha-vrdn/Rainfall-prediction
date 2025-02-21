# # import os
# # import json
# # import joblib
# # import numpy as np
# # import pandas as pd
# # from django.http import JsonResponse
# # from django.conf import settings
# # from django.views.decorators.csrf import csrf_exempt

# # # Load model and label encoder
# # model_path = os.path.join(settings.BASE_DIR, "rainfall_project", "prediction", "models", "rainfall_model.pkl")
# # encoder_path = os.path.join(settings.BASE_DIR, "rainfall_project", "prediction", "models", "label_encoder.pkl")

# # if not os.path.exists(model_path) or not os.path.exists(encoder_path):
# #     raise FileNotFoundError("Model or encoder file not found. Please train and save the model first.")

# # model = joblib.load(model_path)
# # label_encoder = joblib.load(encoder_path)

# # @csrf_exempt
# # def predict_rainfall(request):
# #     """API to predict rainfall for a given subdivision, year, and month."""

# #     if request.method == "GET":
# #         subdivision = request.GET.get("subdivision")
# #         year = request.GET.get("year")
# #         month = request.GET.get("month")

# #     elif request.method == "POST":
# #         try:
# #             data = json.loads(request.body.decode("utf-8"))
# #             subdivision = data.get("subdivision")
# #             year = data.get("year")
# #             month = data.get("month")
# #         except json.JSONDecodeError:
# #             return JsonResponse({"error": "Invalid JSON data"}, status=400)

# #     else:
# #         return JsonResponse({"error": "Invalid request method"}, status=405)

# #     # Validate inputs
# #     if not subdivision or not year or not month:
# #         return JsonResponse({"error": "Missing parameters"}, status=400)

# #     try:
# #         year = int(year)
# #         month_num = pd.to_datetime(month, format='%b').month
# #         sub_encoded = label_encoder.transform([subdivision])[0]
# #     except Exception as e:
# #         return JsonResponse({"error": str(e)}, status=400)

# #     # Predict rainfall
# #     prediction = model.predict(np.array([[sub_encoded, year, month_num]]))[0]

# #     return JsonResponse({
# #         "subdivision": subdivision,
# #         "year": year,
# #         "month": month,
# #         "predicted_rainfall": round(prediction, 2)
# #     })

# # import os
# # import numpy as np
# # import joblib
# # import pandas as pd
# # from django.shortcuts import render
# # from django.conf import settings
# # from models import *

# # # Load model and label encoder
# # model_path = os.path.join(
# #     settings.BASE_DIR, "prediction", "models", "rainfall_model.pkl"
# # )
# # encoder_path = os.path.join(
# #     settings.BASE_DIR, "prediction", "models", "label_encoder.pkl"
# # )
# # model = joblib.load(model_path)
# # label_encoder = joblib.load(encoder_path)


# # def predict_rainfall_view(request):
# #     predicted_rainfall = None

# #     if request.method == "POST":
# #         subdivision = request.POST.get("subdivision", "").strip()
# #         year = request.POST.get("year", "").strip()
# #         month = request.POST.get("month", "").strip()

# #         if subdivision and year and month:
# #             try:
# #                 year = int(year)
# #                 month_num = pd.to_datetime(month, format="%b").month
# #                 sub_encoded = label_encoder.transform([subdivision])[0]

# #                 # Predict rainfall
# #                 prediction = model.predict(np.array([[sub_encoded, year, month_num]]))[
# #                     0
# #                 ]
# #                 predicted_rainfall = round(prediction, 2)

# #             except Exception as e:
# #                 predicted_rainfall = f"Error: {str(e)}"

# #     return render(
# #         request,
# #         "predictions/index.html",
# #         {"predicted_rainfall": predicted_rainfall},
# #     )


# # def predict_rainfall_view(request):
# #     predicted_rainfall = None

# #     model_path = os.path.join(settings.BASE_DIR, "rainfall_project", "prediction", "models", "rainfall_model.pkl")
# #     encoder_path = os.path.join(settings.BASE_DIR, "rainfall_project", "prediction", "models", "label_encoder.pkl")

# #     Extract unique subdivisions from dataset
# #     df = pd.read_csv("Rainfall.csv")  # Load CSV file
# #     subdivisions = sorted(
# #         df["SUBDIVISION"].unique()
# #     )  # Get sorted list of unique subdivisions

# #     years = list(range(1901, 2101))  # Range of years from dataset + future
# #     months = {
# #         "JAN": "January",
# #         "FEB": "February",
# #         "MAR": "March",
# #         "APR": "April",
# #         "MAY": "May",
# #         "JUN": "June",
# #         "JUL": "July",
# #         "AUG": "August",
# #         "SEP": "September",
# #         "OCT": "October",
# #         "NOV": "November",
# #         "DEC": "December",
# #     }

# #     if request.method == "POST":
# #         subdivision = request.POST.get("subdivision", "").strip()
# #         year = request.POST.get("year", "").strip()
# #         month = request.POST.get("month", "").strip()

# #         if subdivision and year and month:
# #             try:
# #                 year = int(year)
# #                 month_num = pd.to_datetime(month, format="%b").month
# #                 sub_encoded = label_encoder.transform([subdivision])[0]

# #                 Predict rainfall
# #                 prediction = model.predict(np.array([[sub_encoded, year, month_num]]))[
# #                     0
# #                 ]
# #                 predicted_rainfall = round(prediction, 2)

# #             except Exception as e:
# #                 predicted_rainfall = f"Error: {str(e)}"

# #     return render(
# #         request,
# #         "predictions/index.html",
# #         {
# #             "predicted_rainfall": predicted_rainfall,
# #             "subdivisions": subdivisions,
# #             "years": years,
# #             "months": months,
# #         },
# #     )


import os
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render

# Load the model and label encoder
model_path = os.path.join(os.path.dirname(__file__), "models/rainfall_model.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "models/label_encoder.pkl")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)


def predict_rainfall_view(request):
    predicted_rainfall = None

    # Extract unique subdivisions from dataset
    df = pd.read_csv("Rainfall.csv")
    subdivisions = sorted(df["SUBDIVISION"].unique())

    years = list(range(2024, 2101))
    months = {
        "JAN": "January",
        "FEB": "February",
        "MAR": "March",
        "APR": "April",
        "MAY": "May",
        "JUN": "June",
        "JUL": "July",
        "AUG": "August",
        "SEP": "September",
        "OCT": "October",
        "NOV": "November",
        "DEC": "December",
    }

    if request.method == "POST":
        subdivision = request.POST.get("subdivision", "").strip()
        year = request.POST.get("year", "").strip()
        month = request.POST.get("month", "").strip()

        if subdivision and year and month:
            try:
                year = int(year)
                month_num = pd.to_datetime(month, format="%b").month
                sub_encoded = label_encoder.transform([subdivision])[0]

                # Predict rainfall
                prediction = model.predict(np.array([[sub_encoded, year, month_num]]))[
                    0
                ]
                predicted_rainfall = round(prediction, 2)

            except Exception as e:
                predicted_rainfall = f"Error: {str(e)}"

    return render(
        request,
        "predictions/index.html",
        {
            "predicted_rainfall": predicted_rainfall,
            "subdivisions": subdivisions,
            "years": years,
            "months": months,
        },
    )
